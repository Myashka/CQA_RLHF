import torch
from torch import nn
import pytorch_lightning as pl
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import SacreBLEUScore
import nltk
import numpy as np


class LitLM(pl.LightningModule):
    def __init__(self, model_name, use_cache, batch_size=8, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, use_cache=use_cache
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.end_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.pad_token_id = self.tokenizer.eos_token_id

        if self.hparams.do_compute_metrics:
            nltk.download("punkt")
            self.rouge = ROUGEScore()
            self.bleu = SacreBLEUScore()
            if self.hparams.do_compute_bertscore:
                self.bertscore = BERTScore(lang="en")

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
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        val_loss = output.loss

        preds = output.logits.argmax(dim=-1)
        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "val_loss",
            val_loss,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if self.hparams.do_compute_metrics:

            pred = torch.cat([output["preds"] for output in outputs], dim=0)
            labels = torch.cat([output["labels"] for output in outputs], dim=0)

            preds = self.tokenizer.batch_decode(pred, skip_special_tokens=True)
            labels = self.tokenizer.batch_decode(
                labels, skip_special_tokens=True)

            if self.hparams.do_compute_bertscore:
                self.bertscore(preds, labels)
                self.log_dict("val_bert_score", self.bertscore)

            self.bleu.update(preds, [labels])
            bleu = self.bleu.compute()
            self.log("val_bleu", bleu, on_step=False,
                     on_epoch=True, sync_dist=True)

            self.rouge.update(preds, labels)
            rouge = self.rouge.compute()

            self.log(
                "val_rouge1_fmeasure",
                rouge["rouge1_fmeasure"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "val_rouge2_fmeasure",
                rouge["rouge2_fmeasure"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "val_rougeL_fmeasure",
                rouge["rougeL_fmeasure"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

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
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=self.hparams.adam_betas,
        )

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [lr_scheduler]

    def generate(self, text: str, device, **kwargs):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = inputs.to(device)
        generated_tokens = self.model.generate(inputs["input_ids"], **kwargs)
        generated_q_a = self.tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True
        )
        return generated_q_a
