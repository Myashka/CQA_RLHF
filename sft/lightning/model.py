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
from torchmetrics.functional import bleu_score
import numpy as np


class LitLM(pl.LightningModule):
    def __init__(
        self,
        model_name,
        learning_rate,
        do_freeze,
        use_cache,
        warmup_steps,
        adam_betas,
        weight_decay,
        max_length,
        batch_size=8,
        *args,
        **kwargs
    ):
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
            self.rouge = ROUGEScore()
            self.bleu = bleu_score()
            if self.hparams.do_compute_bertscore:
                self.bertscore = BERTScore()

        if do_freeze:
            for n, p in self.model.named_parameters():
                if "transformer.h" in n:
                    layer_num = int(n.split(".")[2])
                    if "ln_" not in n and layer_num > 0 and layer_num < 23:
                        p.requires_grad = False

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        output = self.model(**batch)
        self.log(
            "train/loss",
            output.loss,
            logger=True,
        )
        return output.loss

    def compute_metrics(self, predictions, references):
        labels_ids = references
        pred_ids = predictions
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        result_dict = dict()

        rouge_dict = self.rouge(preds=pred_str, target=label_str)
        bleu_metric = self.bleu(preds=pred_str, target=label_str)

        result_dict["bleu"] = bleu_metric

        if self.hparams.do_compute_bertscore:
            bertscore_dict = self.bertscore(
                preds=pred_str,
                target=label_str,
                lang="en",
                max_length=self.hparams.max_length,
            )
            result_dict["bert_precision"] = np.mean(bertscore_dict["precision"])
            result_dict["bert_recall"] = np.mean(bertscore_dict["recall"])
            result_dict["bert_f1"] = np.mean(bertscore_dict["f1"])

            result_dict["rouge1_fmeasure"] = rouge_dict["rouge1_fmeasure"]
            result_dict["rouge2_fmeasure"] = rouge_dict["rouge2_fmeasure"]
            result_dict["rougeL_fmeasure"] = rouge_dict["rougeL_fmeasure"]
            result_dict["rougeL_recall"] = rouge_dict["rougeL_recall"]
            result_dict["rougeL_precision"] = rouge_dict["rougeL_precision"]

        return result_dict

    def validation_step(self, batch, batch_idx):
        # this is the test loop
        output = self.model(**batch)
        val_loss = output.loss

        preds = output.logits.argmax(dim=-1)
        labels = batch["labels"]

        # self.log("val_loss", val_loss)

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.do_compute_metrics:
            preds = torch.cat([x["preds"] for x in outputs]).detach()
            labels = torch.cat([x["labels"] for x in outputs]).detach()
            self.log_dict(
                self.compute_metrics(predictions=preds, references=labels),
                logger=True,
            )
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val/loss", loss, logger=True, prog_bar=True)

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
