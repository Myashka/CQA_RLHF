from tqdm.auto import tqdm
from pathlib import Path
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from evaluate import load
from accelerate.utils import set_seed
import numpy as np

import torch

import time

from itertools import cycle


class Trainer:
    def __init__(
        self,
        eval_every=500,
        learning_rate=1e-5,
        warmup_steps=100,
        gradient_accumulation_steps=1,
        seed=42,
        log_with="all",  # None, 'all', 'tensorboard', 'wandb', 'comet_ml'
        output_dir="./outputs",
        max_grad_norm=None,
        tracker_init_kwargs=None,
        cpu=False,
        resume_from_checkpoint=None,
        accelerator=None,
        **kwargs,
    ):
        self.eval_every = int(eval_every)
        self.learning_rate = float(learning_rate)
        self.warmup_steps = int(warmup_steps)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.seed = int(seed)
        self.log_with = log_with
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.max_grad_norm = max_grad_norm
        self.tracker_init_kwargs = tracker_init_kwargs
        self.cpu = cpu
        self.resume_from_checkpoint = resume_from_checkpoint
        self.__dict__.update(kwargs)

        self.accelerator = accelerator or Accelerator(
            log_with=self.log_with,
            logging_dir=self.output_dir.as_posix(),
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            cpu=self.cpu,
        )
        set_seed(self.seed)

        self.global_step = 0

        self.config = dict(
            eval_every=self.eval_every,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            gradient_accumulation_steps=int(self.gradient_accumulation_steps),
            seed=self.seed,
            max_grad_norm=self.max_grad_norm,
        )

        self.rouge = load("rouge")
        self.bertscore = load("bertscore")
        self.bleu = load("bleu")

    def train(
        self,
        model,
        tokenizer,
        n_epoches,
        train_loader,
        val_loader=None,
        optimizer=None,
        scheduler=None,
    ):

        self.max_steps = n_epoches * len(train_loader)

        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        # self.scheduler = scheduler or get_linear_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_warmup_steps=int(self.warmup_steps),
        #     num_training_steps=int(self.max_steps)
        #     // int(self.gradient_accumulation_steps),
        # )
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.max_steps,
            num_training_steps=(self.max_steps) // self.gradient_accumulation_steps,
        )

        if self.resume_from_checkpoint:
            self.load_checkpoint(self.resume_from_checkpoint)

        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            model, self.optimizer, self.scheduler, train_loader, val_loader
        )

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                "CQA_RLHF",
                config=self.config,
                init_kwargs=self.tracker_init_kwargs or {},
            )

        self.train_data_loader_length = len(self.train_loader)
        self.train_loader = cycle(self.train_loader)
        self.progress_bar = tqdm(
            initial=self.global_step,
            total=int(self.max_steps),
            disable=not self.accelerator.is_main_process,
        )
        while self.global_step < self.max_steps:
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                batch = next(self.train_loader)
                with self.accelerator.accumulate(self.model):
                    outputs = self.train_step(batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    if self.max_grad_norm and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )

                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()

            # Log Train Loss + Learning Rate + Global Step
            self.accelerator.log({"train_loss": loss.item()}, step=self.global_step)
            self.accelerator.log(
                {"lr": self.optimizer.param_groups[0]["lr"]}, step=self.global_step
            )
            self.accelerator.log(
                {"global_step": self.global_step}, step=self.global_step
            )
            self.accelerator.log(
                {"epoch": self.global_step / self.train_data_loader_length},
                step=self.global_step,
            )

            self.global_step += 1
            if self.accelerator.is_main_process:
                self.progress_bar.update(1)
                self.progress_bar.set_description(f"loss {loss.item():.4f}")

            if self.global_step % self.eval_every == 0:
                self.model.eval()
                self.evaluate(self.val_loader, tokenizer)
                self.save_checkpoint()
                self.model.train()

        self.save_checkpoint("last.ckpt")
        self.accelerator.end_training()

    def train_step(self, batch):
        return self.model(**batch)

    def compute_metrics(self, tokenizer, predictions, references):
        labels_ids = references
        pred_ids = predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        result_dict = self.rouge.compute(predictions=pred_str, references=label_str)
        bertscore_dict = self.bertscore.compute(
            predictions=pred_str, references=label_str, lang="en"
        )
        bleu_metric = self.bleu.compute(predictions=pred_str, references=label_str)[
            "bleu"
        ]

        result_dict["bert_precision"] = np.mean(bertscore_dict["precision"])
        result_dict["bert_recall"] = np.mean(bertscore_dict["recall"])
        result_dict["bert_f1"] = np.mean(bertscore_dict["f1"])

        result_dict["bleu"] = bleu_metric

        return result_dict

    def evaluate(self, val_loader, tokenizer):
        self.accelerator.print("\nEvaluating...")
        total_loss = 0
        all_predictions = []
        all_labels = []

        for batch in val_loader:
            with torch.no_grad():
                output = self.val_step(batch)
                loss = output.loss
                predictions = output.logits.argmax(dim=-1)
                all_predictions.append(self.accelerator.gather(predictions))
                all_labels.append(self.accelerator.gather(batch["labels"]))

            total_loss += loss.float()

        self.accelerator.print("\nConcatenating predictions and labels...")
        all_predictions = torch.cat(all_predictions)[
            : int(len(val_loader) * len(batch["input_ids"]))
        ]
        all_labels = torch.cat(all_labels)[
            : int(len(val_loader) * len(batch["input_ids"]))
        ]

        eval_metric = self.compute_metrics(
            tokenizer, predictions=all_predictions, references=all_labels
        )
        self.accelerator.print(f"Metrics computed\n{eval_metric}")

        self.accelerator.log({"val_loss": total_loss.item()}, step=self.global_step)
        self.accelerator.log(
            {"epoch": self.global_step / self.train_data_loader_length},
            step=self.global_step,
        )
        self.accelerator.log(
            {
                "bleu": eval_metric["bleu"],
                "bert_f1": eval_metric["bert_f1"],
                "rouge1": eval_metric["rouge1"],
                "rougeL": eval_metric["rougeL"],
            },
            step=self.global_step,
        )
        self.accelerator.print("Metrics loged")

    def val_step(self, batch):
        return self.model(**batch)

    def save_checkpoint(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            model = self.accelerator.unwrap_model(self.model)
            ckpt_path = str(self.output_dir) + f"/step_{self.global_step}.ckpt"
            save_obj = {
                "model": model.state_dict(),
                "global_step": self.global_step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            }
            self.accelerator.save(save_obj, ckpt_path)
            self.accelerator.print(f"Saved checkpoint to: {ckpt_path}")

    def load_checkpoint(
        self, ckpt_path, strict=True, model_only=False, resume_global_step=True
    ):
        loaded_obj = torch.load(ckpt_path, map_location="cpu")

        self.model.load_state_dict(loaded_obj["model"], strict=strict)

        if not model_only:
            self.optimizer.load_state_dict(loaded_obj["optimizer"])
            self.scheduler.load_state_dict(loaded_obj["scheduler"])
            self.global_step = (
                loaded_obj["global_step"] if resume_global_step else self.global_step
            )

        self.accelerator.print(f"Loaded checkpoint {ckpt_path}")
