import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def prepare_train(
    data_file_path,
    tokenizer,
    splits,
    max_prompt_length=None,
    truncate_promt=True,
    max_length=None,
):
    def promt_tokenize(examples):
        j_a_toks = tokenizer.encode(examples['Accepted_Answer'])
        k_a_toks = tokenizer.encode(examples['Answer'])

        if truncate_promt:
            q_toks = tokenizer.encode(examples['Question'])
            q_toks = q_toks[:max_prompt_length-7]
            tmp = tokenizer.decode(q_toks).strip()
        else:
            tmp = examples['Question']

        tmp = 'Question: ' + tmp + "\nAnswer:"
        q_toks = tokenizer.encode(tmp)

        j_sample = torch.tensor(q_toks + j_a_toks, dtype=int)
        j_sample = j_sample[:max_length]
        j_attention_mask = torch.ones(j_sample.shape, dtype=int)

        k_sample = torch.tensor(q_toks + k_a_toks, dtype=int)
        k_sample = k_sample[:max_length]
        k_attention_mask = torch.ones(k_sample.shape, dtype=int)

        return {'input_ids_j': j_sample, 'attention_mask_j': j_attention_mask, 'input_ids_k': k_sample, 'attention_mask_k': k_attention_mask}

    datasets = []
    for split in splits:
        dataset = load_dataset(
            "json", data_files=f"{data_file_path}", field=f'{split}')['train']
        dataset = dataset.map(promt_tokenize)
        dataset.set_format(type="torch", columns=[
                           "input_ids_j", "attention_mask_j", "input_ids_k", 'attention_mask_k'])
        datasets.append(dataset)
    return datasets


def prepare_inference(data_file_path,
                      tokenizer,
                      split,
                      max_length,
                      max_prompt_length,
                      padding_side,
                      padding,
                      truncate_promt):
    tokenizer.padding_side = padding_side

    def promt_tokenize(examples):

        a_toks_len = len(tokenizer.encode(examples['Answer']))
        if truncate_promt:
            q_toks = tokenizer.encode(examples['Question'])
            q_toks = q_toks[:max_prompt_length - 7]
            tmp = tokenizer.decode(q_toks).strip()
        else:
            tmp = examples['Question']

        tmp = 'Question: ' + tmp + "\nAnswer: " + examples['Answer']

        tokenized_dict = tokenizer(
            [tmp], padding=padding, max_length=max_length, truncation=True)

        return tokenized_dict

    dataset = load_dataset(
        "json", data_files=f"{data_file_path}", field=f'{split}')['train']
    dataset = dataset.map(promt_tokenize)
    dataset.set_format(type="torch", columns=[
                       'Question', 'Answer', 'Score', "input_ids", "attention_mask"])

    return dataset


def prepare_dataloader(dataset, tokenizer, batch_size, shuffle, on_tpu, max_length=None):

    if not on_tpu:
        max_length = None
    else:
        assert max_length is not None, "Be sure max_length is notn None!"

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda data: {
                          "input_ids_j": collate_batch(
                              [f["input_ids_j"] for f in data],
                              tokenizer,
                              max_length=max_length),
                          "attention_mask_j": collate_batch(
                              [f["attention_mask_j"] for f in data],
                              tokenizer,
                              "attention_mask",
                              max_length=max_length
                          ),
                          "input_ids_k": collate_batch(
                              [f["input_ids_k"] for f in data],
                              tokenizer,
                              max_length=max_length),
                          "attention_mask_k": collate_batch(
                              [f["attention_mask_k"] for f in data],
                              tokenizer,
                              "attention_mask",
                              max_length=max_length
                          )
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
