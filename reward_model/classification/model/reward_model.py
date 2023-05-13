import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import (
    GPTNeoForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class GPTneo_Regressor(pl.LightningModule):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = GPTNeoForSequenceClassification.from_pretrained(
            model_name, use_cache=self.hparams.use_cache, num_labels=1
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.end_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.pad_token_id = self.tokenizer.eos_token_id

        self.frozen = False
        self.freeze()

    def training_step(self, batch, batch_idx):
        j_rewards = self.model(input_ids=batch['input_ids_j'], attention_mask=batch['attention_mask_j']).logits.squeeze()
        k_rewards = self.model(input_ids=batch['input_ids_k'], attention_mask=batch['attention_mask_k']).logits.squeeze()
        
        loss = -nn.functional.logsigmoid(j_rewards - k_rewards).mean()

        self.log(
            "train_loss",
            loss,
            logger=True,
            on_step=True,
            sync_dist=True,
        )
        return {'loss': loss, 'j_rewards': j_rewards, 'k_rewards': k_rewards}

    def training_step_end(self, outputs):
        outputs['j_rewards'] = outputs['j_rewards'].reshape((1, -1))[0]
        outputs['k_rewards'] = outputs['k_rewards'].reshape((1, -1))[0]
        train_acc = sum(outputs['j_rewards'] > outputs['k_rewards']).item()/len(outputs['k_rewards'])
        self.log('train_accuracy', train_acc,
                 on_step=True, on_epoch=False, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        j_rewards = self.model(input_ids=batch['input_ids_j'], attention_mask=batch['attention_mask_j']).logits.squeeze()
        k_rewards = self.model(input_ids=batch['input_ids_k'], attention_mask=batch['attention_mask_k']).logits.squeeze()
        
        val_loss = -nn.functional.logsigmoid(j_rewards - k_rewards).mean()

        self.log(
            "val_loss",
            val_loss,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return {'loss': val_loss, 'j_rewards': j_rewards, 'k_rewards': k_rewards}

    def validation_step_end(self, outputs):
        outputs['j_rewards'] = outputs['j_rewards'].reshape((1, -1))[0]
        outputs['k_rewards'] = outputs['k_rewards'].reshape((1, -1))[0]
        val_acc = sum(outputs['j_rewards'] > outputs['k_rewards']).item()/len(outputs['k_rewards'])

        self.log('val_accuracy', val_acc, on_step=False,
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
            num_warmup_steps=self.hparams.warmup_steps_per_cent *
            self.trainer.estimated_stepping_batches,
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