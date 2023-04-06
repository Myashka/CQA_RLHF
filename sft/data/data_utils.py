import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader


def prepare_train(
    data_file_path,
    tokenizer,
    splits,
    max_length=None,
    truncate_promt=True,
):
    def promt_tokenize(examples):
        a_toks = tokenizer.encode(examples['Answer'])

        if truncate_promt:
            q_toks = tokenizer.encode(examples['Question'])
            q_toks = q_toks[:max_length - len(a_toks)-7]
            tmp = tokenizer.decode(q_toks).strip()
        else:
            tmp = examples['Question']

        tmp = 'Question: ' + tmp + "\nAnswer:"
        q_toks = tokenizer.encode(tmp)

        sample = torch.tensor(q_toks + a_toks, dtype=int)
        sample = sample[:max_length]
        attention_mask = torch.ones(sample.shape, dtype=int)
        labels = sample.clone()
        labels[:len(q_toks)] = -100

        return {'input_ids': sample, 'attention_mask': attention_mask, 'labels': labels}

    datasets = []
    for split in splits:
        dataset = load_dataset(
            "json", data_files=f"{data_file_path}", field=f'{split}')['train']
        dataset = dataset.map(promt_tokenize)
        dataset.set_format(type="torch", columns=[
                           "input_ids", "attention_mask", "labels"])
        datasets.append(dataset)
    return datasets


def prepare_inference(data_file_path,
                      tokenizer,
                      split,
                      max_length,
                      padding_side,
                      padding):
    tokenizer.padding_side = padding_side

    def promt_tokenize(examples):
        q_toks = tokenizer.encode(examples['Question'])
        q_toks = q_toks[: max_length-7]
        tmp = tokenizer.decode(q_toks).strip()

        tmp = 'Question: ' + tmp + "\nAnswer:"

        tokenized_dict = tokenizer(
            tmp, padding=padding, max_length=max_length, truncation=True)

        return tokenized_dict

    dataset = load_dataset(
        "json", data_files=f"{data_file_path}", field=f'{split}')['train']
    dataset = dataset.map(promt_tokenize)
    dataset.set_format(type="torch", columns=[
                       'Question', 'Answer', "input_ids", "attention_mask"])

    return dataset


def prepare_dataloader_with_labels(dataset, tokenizer, batch_size, shuffle, on_tpu, max_length=None):

    if not on_tpu:
        max_length = None
    else:
        assert max_length is not None, "Be sure max_length is notn None!"

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda data: {
                          "input_ids": collate_batch(
                              [f["input_ids"] for f in data],
                              tokenizer,
                              max_length=max_length),
                          "attention_mask": collate_batch(
                              [f["attention_mask"] for f in data],
                              tokenizer,
                              "attention_mask",
                              max_length=max_length
                          ),
                          "labels": collate_batch(
                              [f["labels"] for f in data],
                              tokenizer,
                              max_length=max_length
                          ),
                      },
                      )


def collate_batch(examples, tokenizer, input_type="input_ids", max_length=None):

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(
        x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    if not max_length:
        max_length = max(x.size(0) for x in examples)

    if input_type == "input_ids":
        result = examples[0].new_full(
            [len(examples), max_length], tokenizer.pad_token_id
        )
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0]:] = example
    elif input_type == "attention_mask":
        result = examples[0].new_full([len(examples), max_length], 0)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0]:] = example
    return result
