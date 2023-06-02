import json

import torch
from torch.utils.data import Dataset


class QADataset(Dataset):
    def __init__(
        self,
        file_name,
        tokenizer,
        split="train",
        max_length=256,
        truncate_promt=True,
        train=True,
    ):
        self.train = train
        self.pairs = []
        with open(file_name, "r") as f:
            data = json.load(f)
            self.pairs = data[split]

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncate_promt = truncate_promt

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        qa_pair = self.pairs[idx]
        if not self.train:
            return "Question: " + qa_pair["Question"], qa_pair["Answer"]
        else:

            sample, attention_mask, labels = self.promt_tokenize(
                qa_pair["Question"], qa_pair["Answer"]
            )

            return {
                "input_ids": sample,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    def promt_tokenize(self, promt, completion):
        a_toks = self.tokenizer.encode(completion)

        if self.truncate_promt:
            q_toks = self.tokenizer.encode(promt)
            q_toks = q_toks[: self.max_length - len(a_toks) - 7]
            tmp = self.tokenizer.decode(q_toks).strip()
        else:
            tmp = promt

        tmp = "Question: " + tmp + "\nAnswer:"
        q_toks = self.tokenizer.encode(tmp)

        sample = torch.tensor(q_toks + a_toks, dtype=int).unsqueeze(0)
        sample = sample[0, : self.max_length]
        attention_mask = torch.ones(sample.shape, dtype=int)
        labels = sample.clone()
        labels[0, : len(q_toks)] = -100

        return sample, attention_mask, labels
