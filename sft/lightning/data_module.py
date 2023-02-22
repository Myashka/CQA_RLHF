import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os

os.path.append(r"D:\CQA_RLHF\sft\dataset")
from dataset import dataset, data_utils


class QADataModule(pl.LightningDataModule):
    def __init__(self, model_name, data_dir, max_length, batch_size, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.pairs = []

    def prepare_data(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_ds, self.val_ds = dataset.prepare_datasets(
                self.hparams.data_dir,
                self.tokenizer,
                max_length=self.hparams.max_length,
                splits=["train", "val"],
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_ds = dataset.prepare_datasets(
                self.hparams.data_dir,
                self.tokenizer,
                max_length=self.hparams.max_length,
                splits=["test"],
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=lambda data: {
                "input_ids": data_utils.collate_batch(
                    [f["input_ids"] for f in data],
                    self.tokenizer,
                    max_length=self.hparams.max_length,
                ),
                "attention_mask": data_utils.collate_batch(
                    [f["attention_mask"] for f in data],
                    self.tokeizer,
                    "attention_mask",
                    max_length=self.hparams.max_length,
                ),
                "labels": data_utils.collate_batch(
                    [f["labels"] for f in data],
                    self.tokenizer,
                    max_length=self.hparams.max_length,
                ),
            },
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=lambda data: {
                "input_ids": data_utils.collate_batch(
                    [f["input_ids"] for f in data],
                    self.tokenizer,
                    max_length=self.hparams.max_length,
                ),
                "attention_mask": data_utils.collate_batch(
                    [f["attention_mask"] for f in data],
                    self.tokeizer,
                    "attention_mask",
                    max_length=self.hparams.max_length,
                ),
                "labels": data_utils.collate_batch(
                    [f["labels"] for f in data],
                    self.tokenizer,
                    max_length=self.hparams.max_length,
                ),
            },
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=lambda data: {
                "input_ids": data_utils.collate_batch(
                    [f["input_ids"] for f in data],
                    self.tokenizer,
                    max_length=self.hparams.max_length,
                ),
                "attention_mask": data_utils.collate_batch(
                    [f["attention_mask"] for f in data],
                    self.tokeizer,
                    "attention_mask",
                    max_length=self.hparams.max_length,
                ),
                "labels": data_utils.collate_batch(
                    [f["labels"] for f in data],
                    self.tokenizer,
                    max_length=self.hparams.max_length,
                ),
            },
        )
