import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_utils import collate_batch
import json


class QADataset(Dataset):
    def __init__(
        self,
        file_name,
        tokenizer,
        split="train",
        max_length=None,
        padding=False,
        zero_question_labels=False,
    ):
        self.pairs = []
        with open(file_name, "r") as f:
            data = json.load(f)
            self.pairs = data[split]

        self.tokenizer = tokenizer
        self.zero_question_labels = zero_question_labels
        self.max_length = max_length
        self.padding = padding

        if self.max_length is None:

            self.max_length = 0
            for pair in self.pairs:
                sample_length = len(
                    self.tokenizer.encode(
                        r"Question\n"
                        + pair["Title"]
                        + ". "
                        + pair["Question"]
                        + r"\nAnswer:"
                        + pair["Answer"],
                        return_tensors="pt",
                    )[0]
                )

                if sample_length > self.max_length:
                    self.max_length = sample_length

                if self.max_length > 2048:
                    self.max_length = 2048

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        qa_pair = self.pairs[idx]
        tokenized_dict = self.tokenizer(
            r"Question\n" + qa_pair["Question"] + r"\nAnswer:" + qa_pair["Answer"],
            truncation=True,
            max_length=self.max_length,
            padding=self.padding,
            return_tensors="pt",
        )

        if self.zero_question_labels:
            question_len = len(self.tokenizer.encode(qa_pair["Question"]))
            labels = tokenized_dict["input_ids"].clone()
            labels[-1][:question_len] = -100
        else:
            labels = tokenized_dict["input_ids"].clone()

        return {
            "input_ids": tokenized_dict["input_ids"][0],
            "attention_mask": tokenized_dict["attention_mask"][0],
            "labels": labels[0],
        }


def prepare_datasets(
    data_file_path,
    tokenizer,
    splits,
    max_length=None,
    padding=False,
    zero_question_labels=False,
):
    datasets = []
    for split in splits:
        datasets.append(
            QADataset(
                data_file_path,
                tokenizer,
                split=split,
                max_length=max_length,
                padding=padding,
                zero_question_labels=zero_question_labels,
            )
        )
    return datasets


def create_dataloaders(
    data_file_path,
    tokenizer,
    splits,
    batch_sizes,
    max_length=512,
    zero_questions_labels=False,
    all_max_length=None,
):
    datasets = prepare_datasets(
        data_file_path, tokenizer, splits, max_length, zero_questions_labels
    )

    dataloaders = []

    if all_max_length:
        same_max_length = max_length
    else:
        same_max_length = None
    for dataset, batch_size in zip(datasets, batch_sizes):
        dataloaders.append(
            DataLoader(
                dataset,
                shuffle=True,
                batch_size=batch_size,
                collate_fn=lambda data: {
                    "input_ids": collate_batch(
                        [f["input_ids"] for f in data],
                        tokenizer,
                        max_length=same_max_length,
                    ),
                    "attention_mask": collate_batch(
                        [f["attention_mask"] for f in data],
                        tokenizer,
                        "attention_mask",
                        max_length=same_max_length,
                    ),
                    "labels": collate_batch(
                        [f["labels"] for f in data],
                        tokenizer,
                        max_length=same_max_length,
                    ),
                },
            )
        )

    return dataloaders
