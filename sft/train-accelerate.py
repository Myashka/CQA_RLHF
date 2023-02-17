from dataset.dataset import prepare_datasets
from dataset.data_utils import collate_batch
import yaml
from yaml import CLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

import torch
import os
from evaluate import load
import numpy as np
import click
import random
from warnings import filterwarnings
import wandb

from accelerate import Accelerator
from accelerate.utils import set_seed, find_executable_batch_size

@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file, model):
    accelerator = Accelerator()
    
    def training_function(model):
        with open("config.yaml", "r") as f:
            config = yaml.load(f, Loader=CLoader)

        set_seed(config["random_seed"])

        if config["try_cuda"]:
            device = torch.device("cuda") if torch.cuda.is_available else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model.resize_token_embeddings(len(tokenizer))
        tokenizer.pad_token = tokenizer.eos_token
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model.pad_token_id = tokenizer.eos_token_id

        rouge = load("rouge")
        bertscore = load("bertscore")
        bleu = load("bleu")

        data_file_path = config["data"]["file_path"]

        train_dataset, val_dataset = prepare_datasets(
            data_file_path,
            tokenizer,
            ["train", "val"],
            config["data"]["max_length"],
            config["data"]["zero_question"],
        )

        def compute_metrics(eval_preds):
            labels_ids = eval_preds.label_ids
            pred_ids = eval_preds.predictions
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

            result_dict = rouge.compute(predictions=pred_str, references=label_str)
            bertscore_dict = bertscore.compute(
                predictions=pred_str, references=label_str, lang="en"
            )
            bleu_metric = bleu.compute(predictions=pred_str, references=label_str)["bleu"]

            result_dict["bert_precision"] = np.mean(bertscore_dict["precision"])
            result_dict["bert_recall"] = np.mean(bertscore_dict["recall"])
            result_dict["bert_f1"] = np.mean(bertscore_dict["f1"])

            result_dict["bleu"] = bleu_metric

            return result_dict

        # Create a preprocessing function to extract out the proper logits from the model output
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        # Freeze the model
        if config["train"]["do_freeze"]:
            for n, p in model.named_parameters():
                if "transformer.h" in n:
                    layer_num = int(n.split(".")[2])
                    if "ln_" not in n and layer_num > 0 and layer_num < 23:
                        p.requires_grad = False
            print("Model freezeds")

        wandb.login()
        run = wandb.init(
            project=config["wandb_project"],
            entity="myashka",
            config=config,
            job_type="train",
            group="sft",
        )

        os.environ["WANDB_LOG_MODEL"] = "true"
        os.environ["WANDB_DISABLED"] = "true"

        training_args = TrainingArguments(
            output_dir=config["train"]["checkpoint_dir"],
            num_train_epochs=config["train"]["num_epochs"],
            learning_rate=config["train"]["learning_rate"],
            per_device_train_batch_size=config["train"]["train_batch_size"],
            per_device_eval_batch_size=config["train"]["eval_batch_size"],
            gradient_checkpointing=True,
            half_precision_backend=True,
            fp16=True,
            adam_beta1=0.9,
            adam_beta2=0.95,
            gradient_accumulation_steps=config["train"]["gradient_accumulation_steps"],
            warmup_steps=config["train"]["warmup_steps"],
            evaluation_strategy="steps",
            eval_accumulation_steps=1,
            eval_steps=config["train"]["eval_steps"],
            save_steps=config["train"]["save_steps"],
            load_best_model_at_end=True,
            logging_steps=config["train"]["logging_steps"],
            report_to="wandb",
            run_name=config["train"]["wandb_run_name"],
            # deepspeed=config["train"]["ds_config_file"],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=lambda data: {
                "input_ids": collate_batch([f["input_ids"] for f in data], tokenizer),
                "attention_mask": collate_batch(
                    [f["attention_mask"] for f in data], tokenizer, "attention_mask"
                ),
                "labels": collate_batch([f["labels"] for f in data], tokenizer),
            },
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        trainer = accelerator.prepare(trainer)
        trainer.train()
        trainer.save_model(config["train"]["checkpoint_dir"])
        wandb.finish()
    training_function(model)


if __name__ == "__main__":
    filterwarnings("ignore")
    model_name = 'EleutherAI/gpt-neo-1.3B'
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)
    main(model=model)
