import torch
import pytorch_lightning as pl
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import SacreBLEUScore
import nltk
import gc


class LitLM(pl.LightningModule):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, use_cache=self.hparams.use_cache
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))
        # self.model.config.end_token_id = self.tokenizer.eos_token_id
        # self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.pad_token_id = self.tokenizer.eos_token_id

        if self.hparams.get("do_compute_metrics"):
            nltk.download("punkt")
            self.rouge = ROUGEScore()
            self.bleu = SacreBLEUScore()

        self.frozen = False
        if self.hparams.get("do_freeze"):
            self.freeze()

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
        labels = batch["input_ids"]

        if self.hparams.do_compute_metrics:
            preds = self.tokenizer.batch_decode(
                preds, skip_special_tokens=True)
            labels = self.tokenizer.batch_decode(
                labels, skip_special_tokens=True)

            self.bleu.update(preds, [labels])
            self.rouge.update(preds, labels)

        return {"loss": val_loss}

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

            bleu = self.bleu.compute()
            self.log("val_bleu", bleu, on_step=False,
                     on_epoch=True, sync_dist=True)
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
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=self.hparams.adam_betas,
        )

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(
                self.hparams.warmup_steps_per_cent * self.trainer.estimated_stepping_batches),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1}]

    def freeze(self):
        for name, p in self.model.named_parameters():
            name = name.lower()
            if 'transformer.h' in name and int(name.split('.')[2]) in self.hparams.layers_not_to_freeze:
                continue
            if 'ln' in name or 'norm' in name:
                p.requires_grad = not self.hparams.freeze_ln
            elif 'wte' in name or 'wpe' in name:
                p.requires_grad = not self.hparams.freeze_emb
            elif 'mlp' in name:
                p.requires_grad = not self.hparams.freeze_ff
            elif 'attn' in name:
                p.requires_grad = not self.hparams.freeze_attn
            else:
                p.requires_grad = not self.hparams.freeze_other
            
        self.frozen = True
        print('Model freezed')

    # def freeze(self) -> None:
    #     # freeze all layers, except the final classifier layers
    #     for n, p in self.model.named_parameters():
    #         if "transformer.h" in n:
    #             layer_num = int(n.split(".")[2])
    #             if "ln_" not in n and layer_num > 0 and layer_num < 23:
    #                 p.requires_grad = False

    #     self._frozen = True
    #     print('Model freezed')

    # def unfreeze(self) -> None:
    #     for n, p in self.model.named_parameters():
    #         if "transformer.h" in n:
    #             layer_num = int(n.split(".")[2])
    #             if "ln_" not in n and layer_num > 0 and layer_num < 23:
    #                 p.requires_grad = True

    #     self._frozen = False
    #     print('Model unfreezed')

    # def on_train_epoch_start(self):
    #     """pytorch lightning hook"""
    #     if (self.current_epoch < self.hparams.nr_frozen_epochs) and not self._frozen:
    #         self.freeze()

    #     if (self.current_epoch >= self.hparams.nr_frozen_epochs) and self._frozen:
    #         self.unfreeze()

    def generate(self, input_ids, attention_mask, device, **kwargs):
        gen_input = {'input_ids': input_ids.unsqueeze(0).to(device),
                     'attention_mask': attention_mask.unsqueeze(0).to(device)
                     }
        generated_tokens = self.model.generate(**gen_input, **kwargs)
        generated_q_a = self.tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True
        )
        return generated_q_a
