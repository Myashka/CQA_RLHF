import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from data_utils import collate_batch
import json


class QA_Reward_Dataset(Dataset):
    def __init__(
        self,
        file_name,
        tokenizer,
        split="train",
        max_length=None,
        padding=False,
    ):
        self.pairs = []
        with open(file_name, "r") as f:
            data = json.load(f)
            self.pairs = data[split]

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

        # Определение max_length целой последовательности для оценки
        if self.max_length is None:

            self.max_length = 0
            for pair in self.pairs:
                sample_length = len(
                    self.tokenizer.encode("Question\n" + pair["Question"] + "\nAnswer:"+ pair["Answer"],
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
            "Question\n" + qa_pair["Question"] + "\nAnswer: " + qa_pair["Answer"],
            truncation=True,
            max_length=self.max_length,
            padding=self.padding,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized_dict["input_ids"][0],
            "attention_mask": tokenized_dict["attention_mask"][0],
            "labels": qa_pair['Score'],
        }