from tqdm.auto import tqdm
from pathlib import Path
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from evaluate import load
from accelerate.utils import set_seed, DummyOptim, DummyScheduler
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
        max_length=None,
        tpu=None,
        batch_size=None,
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
        self.max_length = max_length
        self.is_tpu = tpu
        self.batch_size = batch_size
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
            max_length=max_length,
            is_tpu=self.is_tpu,
            batch_size=self.batch_size,
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

        self.max_steps = int(
            n_epoches * len(train_loader) // self.gradient_accumulation_steps
        )
        starting_epoch = 0

        # self.model = model
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "optimizer"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.learning_rate
            )
        else:
            self.optimizer = DummyOptim(model.parameters(), lr=self.learning_rate)

        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.max_steps,
                num_training_steps=(self.max_steps) // self.gradient_accumulation_steps,
            )
        else:
            self.scheduler = DummyScheduler(
                self.optimizer,
                total_num_steps=(self.max_steps) // self.gradient_accumulation_steps,
                warmup_num_steps=self.max_steps,
            )

        if self.resume_from_checkpoint:
            _ = self.load_checkpoint(
                self.resume_from_checkpoint,
                model,
                **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
            )

            resume_step = self.global_step
            starting_epoch = self.global_step // len(train_loader)
            resume_step -= starting_epoch * len(train_loader)

        (
            model,
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

        self.progress_bar = tqdm(
            initial=self.global_step,
            total=int(self.max_steps),
            disable=not self.accelerator.is_main_process,
        )
        for epoch in range(starting_epoch, n_epoches):
            for step, batch in enumerate(self.train_loader):
                if self.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        self.global_step += 1
                        continue
                with self.accelerator.accumulate(model):
                    self.optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    if self.max_grad_norm and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            model.parameters(), self.max_grad_norm
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
                    {"epoch": epoch},
                    step=self.global_step,
                )

                self.global_step += 1
                if self.accelerator.is_main_process:
                    self.progress_bar.update(1)
                    self.progress_bar.set_description(f"loss {loss.item():.4f}")

                if self.global_step % self.eval_every == 0:
                    model.eval()
                    self.evaluate(model, self.val_loader, tokenizer, epoch)
                    self.save_checkpoint(model, epoch)
                    model.train()

        self.save_checkpoint(model, epoch)
        self.accelerator.end_training()

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

    def evaluate(self, model, val_loader, tokenizer, epoch):
        self.accelerator.print("\nEvaluating...")
        losses = []
        all_predictions = []
        all_labels = []

        for batch in val_loader:
            with torch.no_grad():
                output = model(**batch)

            loss = output.loss
            predictions = output.logits.argmax(dim=-1)
            all_predictions.append(self.accelerator.gather(predictions))
            all_labels.append(self.accelerator.gather(batch["labels"]))

            losses.append(
                self.accelerator.gather_for_metrics(
                    loss.repeat(len(batch["input_ids"]))
                )
            )

        losses = torch.cat(losses)
        self.accelerator.print("Concatenating predictions and labels...")
        all_predictions = torch.cat(all_predictions)[
            : int(len(val_loader) * len(batch["input_ids"]))
        ]
        all_labels = torch.cat(all_labels)[
            : int(len(val_loader) * len(batch["input_ids"]))
        ]

        eval_loss = torch.mean(losses)
        self.accelerator.log({"val_loss": eval_loss.item()}, step=self.global_step)
        eval_metric = self.compute_metrics(
            tokenizer, predictions=all_predictions, references=all_labels
        )
        self.accelerator.print(f"Metrics computed\n{eval_metric}")

        self.accelerator.log(
            {
                "bleu": eval_metric["bleu"],
                "bert_f1": eval_metric["bert_f1"],
                "rouge1": eval_metric["rouge1"],
                "rougeL": eval_metric["rougeL"],
                "epoch": epoch,
            },
            step=self.global_step,
        )
        self.accelerator.print("Metrics loged")

    def save_checkpoint(self, model, epoch):
        self.accelerator.wait_for_everyone()
        if self.accelerator.state.deepspeed_plugin is not None:
            if self.accelerator.is_main_process:
                ckpt_path = str(self.output_dir) + f"/step_{self.global_step}.ckpt"
                checkpoint_state_dict = {
                    "epoch": epoch,
                    "last_global_step": self.global_step,
                }
                success = model.save_checkpoint(ckpt_path, epoch, checkpoint_state_dict)
                self.accelerator.print(f"Saved checkpoint to: {ckpt_path}: {success}")
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(model)
            self.accelerator.save(save_obj, ckpt_path)
            self.accelerator.print(f"Saved checkpoint to: {ckpt_path}")
        else:
            if self.accelerator.is_main_process:
                unwrapped_model = self.accelerator.unwrap_model(model)
                ckpt_path = str(self.output_dir) + f"/step_{self.global_step}.ckpt"
                save_obj = {
                    "model": unwrapped_model.state_dict(),
                    "global_step": self.global_step,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "epoch": epoch,
                }
                self.accelerator.save(save_obj, ckpt_path)
                self.accelerator.print(f"Saved checkpoint to: {ckpt_path}")

    def load_checkpoint(
        self,
        ckpt_path,
        model,
        strict=True,
        model_only=False,
        resume_global_step=True,
        **kwargs,
    ):
        if self.accelerator.state.deepspeed_plugin is not None:
            _, checkpoint_state_dict = model.load_checkpoint(ckpt_path, **kwargs)
            epoch = checkpoint_state_dict["epoch"]
            self.global_step = checkpoint_state_dict["last_global_step"]

            del checkpoint_state_dict
            self.accelerator.print(f"Loaded checkpoint {ckpt_path}")
            return epoch
        else:
            loaded_obj = torch.load(ckpt_path, map_location="cpu")

            model.load_state_dict(loaded_obj["model"], strict=strict)

            if not model_only:
                self.optimizer.load_state_dict(loaded_obj["optimizer"])
                self.scheduler.load_state_dict(loaded_obj["scheduler"])
                self.global_step = (
                    loaded_obj["global_step"]
                    if resume_global_step
                    else self.global_step
                )
                epoch = loaded_obj["epoch"]

            self.accelerator.print(f"Loaded checkpoint {ckpt_path}")
            return epoch
