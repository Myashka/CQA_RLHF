import torch
from torch import nn
import pytorch_lightning as pl
from transformers import (
    GPTNeoForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torchmetrics.classification import BinaryAccuracy
import numpy as np


class GPTneo_Regressor(pl.LightningModule):
    def __init__(self, model_name, use_cache, batch_size=8, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = GPTNeoForSequenceClassification.from_pretrained(
            model_name, use_cache=use_cache, num_labels=1
        )

        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.end_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.pad_token_id = self.tokenizer.eos_token_id

        self.train_acc = BinaryAccuracy()

        if self.hparams.do_compute_metrics:
            self.val_acc = BinaryAccuracy()

        if self.hparams.do_freeze:
            for n, p in self.model.named_parameters():
                if "transformer.h" in n:
                    layer_num = int(n.split(".")[2])
                    if "ln_" not in n and layer_num > 0 and layer_num < 23:
                        p.requires_grad = False

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log(
            "train_loss",
            output.loss,
            logger=True,
            on_step=True,
            sync_dist=True,
        )

        preds = (output.logits >= 0).int()
        y = (batch['labels'] >= 0).int()
        return {'loss': output.loss, 'preds': preds, 'target': y}

    def training_step_end(self, outputs):

        outputs['preds'] = outputs['preds'].reshape((1, -1))[0]
        outputs['target'] = outputs['target'].reshape((1, -1))[0]
        self.train_acc(outputs['preds'], outputs['target'])
        self.log('train_accuracy', self.train_acc, on_step=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        val_loss = output.loss

        self.log(
            "val_loss",
            val_loss,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        preds = (output.logits >= 0).int()
        y = (batch['labels'] >= 0).int()

        return {'loss': val_loss, 'preds': preds, 'target': y}

    def validation_step_end(self, outputs):
        if self.hparams.do_compute_metrics:
            outputs['preds'] = outputs['preds'].reshape((1, -1))[0]
            outputs['target'] = outputs['target'].reshape((1, -1))[0]
            self.val_acc(outputs['preds'], outputs['target'])
            self.log('val_accuracy', self.val_acc, on_step=False,
                     on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=self.hparams.adam_betas,
        )

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps_per_cent * self.trainer.estimated_stepping_batches,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [lr_scheduler]
    
    def freeze(self) -> None:
        # freeze all layers, except the final classifier layers
        for n, p in self.model.named_parameters():
                if "transformer.h" in n:
                    layer_num = int(n.split(".")[2])
                    if "ln_" not in n and layer_num > 0 and layer_num < 23:
                        p.requires_grad = False

        self._frozen = True
        print('Model freezed')

    def unfreeze(self) -> None:
        if self._frozen:
            for n, p in self.model.named_parameters():
                if "transformer.h" in n:
                    layer_num = int(n.split(".")[2])
                    if "ln_" not in n and layer_num > 0 and layer_num < 23:
                        p.requires_grad = True

            self._frozen = False
            print('Model unfreezed')

    def on_train_epoch_start(self):
        """pytorch lightning hook"""
        if self.current_epoch < self.hparams.nr_frozen_epochs:
            self.freeze()

        if self.current_epoch >= self.hparams.nr_frozen_epochs:
            self.unfreeze()
