import torch
import numpy as np
from torch.utils.data import Dataset
import json


class QADataset(Dataset):
    def __init__(
        self,
        file_name,
        tokenizer,
        split="train",
        max_length=None,
        zero_question_labels=False,
    ):
        self.pairs = []
        with open(file_name, "r") as f:
            data = json.load(f)
            self.pairs = data[split]

        self.tokenizer = tokenizer
        self.zero_question_labels = zero_question_labels
        self.max_length = max_length

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
            return_tensors="pt",
        )

        if self.zero_question_labels:
            question_len = len(self.tokenizer.encode(qa_pair["Question"])[0])
            labels = tokenized_dict["input_ids"].clone()
            labels[-1][:question_len] = -100
        else:
            labels = tokenized_dict["input_ids"]

        return {
            "input_ids": tokenized_dict["input_ids"][0],
            "attention_mask": tokenized_dict["attention_mask"][0],
            "labels": labels[0],
        }


def prepare_datasets(
    data_file_path,
    tokenizer,
    splits: list[str],
    max_length=None,
    zero_question_labels=True,
):
    datasets = []
    for split in splits:
        datasets.append(
            QADataset(
                data_file_path,
                tokenizer,
                split=split,
                max_length=max_length,
                zero_question_labels=zero_question_labels,
            )
        )
    return datasets