import sys

sys.path.append("/content/CQA_RLHF/sft/dataset")
import argparse
from dataset import create_dataloaders
from accelerate_trainer import Trainer
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
import yaml
from yaml import CLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="If passed, will train on the CPU."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default="wandb",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=".",
        help="Directory with JSON data file.",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="File with DeepSpeed config file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/gpt-neo-1.3B",
        help="Name of model.",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="If passed, will specify use_cache True for model.",
    )

    args = parser.parse_args()
    return args


def main(model, args):

    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
        logging_dir=args.output_dir,
        cpu=args.cpu,
        # downcast_bf16=True if config['TPU'] else False,
    )

    if accelerator.is_main_process:
        wandb.login(key=config["wandb_api"])
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model.resize_token_embeddings(len(tokenizer))
        tokenizer.pad_token = tokenizer.eos_token
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model.pad_token_id = tokenizer.eos_token_id

        if config["freeze"]:
            for n, p in model.named_parameters():
                if "transformer.h" in n:
                    layer_num = int(n.split(".")[2])
                    if "ln_" not in n and layer_num > 0 and layer_num < 23:
                        p.requires_grad = False
            accelerator.print("Model freezeds")

        train_loader, val_loader = create_dataloaders(
            args.data_path,
            tokenizer,
            splits=["train", "val"],
            batch_sizes=[config["batch_size"], config["batch_size"]],
            max_length=config["max_length"],
            all_max_length=config["TPU"],
        )

        trainer = Trainer(
            eval_every=config["eval_every"],
            learning_rate=config["learning_rate"],
            warmup_steps=config["warmup_steps"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            seed=config["seed"],
            log_with=args.log_with,  # None, 'all', 'tensorboard', 'wandb', 'comet_ml'
            output_dir=args.output_dir,
            max_grad_norm=None,
            tracker_init_kwargs=config["wandb_kwargs"],
            cpu=args.cpu,
            resume_from_checkpoint=args.resume_from_checkpoint,
            accelerator=accelerator,
            max_length=config["max_length"],
            tpu=config["TPU"],
            batch_size=config["batch_size"],
        )

        n_epoches = int(config["n_epoches"])
    trainer.train(model, tokenizer, n_epoches, train_loader, val_loader)


if __name__ == "main":
    args = parse_args()
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=args.use_cache)
    main(model, args)