import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import dataset
import data_utils
import torch


class QA_Reward_DataModule(pl.LightningDataModule):
    def __init__(
        self, model_name, data_dir, max_length, batch_size, pad_for_tpu, *args, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pairs = []
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if pad_for_tpu:
            self.padding = "max_length"
        else:
            self.padding = False

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_ds = dataset.QA_Reward_Dataset(
                self.hparams.data_dir, self.tokenizer, max_length=self.hparams.max_length, padding=self.padding)
            self.val_ds = dataset.QA_Reward_Dataset(
                self.hparams.data_dir, self.tokenizer, max_length=self.hparams.max_length, padding=self.padding, split='val')

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_ds = dataset.QA_Reward_Dataset(
                self.hparams.data_dir, self.tokenizer, max_length=self.hparams.max_length, padding=self.padding, split='test')

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=lambda data: {
                "input_ids": data_utils.collate_batch(
                    [f["input_ids"] for f in data],
                    self.tokenizer,
                ),
                "attention_mask": data_utils.collate_batch(
                    [f["attention_mask"] for f in data],
                    self.tokenizer,
                    "attention_mask",
                ),
                "labels": torch.tensor([f["labels"] for f in data])
            },
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda data: {
                "input_ids": data_utils.collate_batch(
                    [f["input_ids"] for f in data],
                    self.tokenizer,
                ),
                "attention_mask": data_utils.collate_batch(
                    [f["attention_mask"] for f in data],
                    self.tokenizer,
                    "attention_mask",
                ),
                "labels": torch.tensor([f["labels"] for f in data])
            },
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda data: {
                "input_ids": data_utils.collate_batch(
                    [f["input_ids"] for f in data],
                    self.tokenizer,
                ),
                "attention_mask": data_utils.collate_batch(
                    [f["attention_mask"] for f in data],
                    self.tokenizer,
                    "attention_mask",
                ),
                "labels": torch.tensor([f["labels"] for f in data])
            },
        )
