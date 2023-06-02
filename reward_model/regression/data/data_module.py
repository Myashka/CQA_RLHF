import pytorch_lightning as pl
from data_utils import (prepare_dataloader_with_labels, prepare_inference,
                        prepare_train)
from transformers import AutoTokenizer


class QA_Reward_DataModule(pl.LightningDataModule):
    def __init__(
        self, model_name, *args, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pairs = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_ds, self.val_ds = prepare_train(
                self.hparams.data_dir,
                self.tokenizer,
                max_length=self.hparams.max_length,
                truncate_promt=self.hparams.truncate_promt,
                splits=["train", "val"],
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_ds = prepare_inference(
                self.hparams.data_dir,
                self.tokenizer,
                max_length=self.hparams.max_length,
                split=self.hparams.split,
                padding_side=self.hparams.padding_side,
                padding=self.hparams.padding,
                truncate_promt=self.hparams.truncate_promt,
            )

    def train_dataloader(self):
        dataloader = prepare_dataloader_with_labels(
            self.train_ds, self.tokenizer, self.hparams.batch_size, True, self.hparams.on_tpu, self.hparams.max_length)
        return dataloader

    def val_dataloader(self):
        dataloader = prepare_dataloader_with_labels(
            self.val_ds, self.tokenizer, self.hparams.batch_size, False, self.hparams.on_tpu, self.hparams.max_length)
        return dataloader
