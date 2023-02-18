import sys

sys.path.append("/content/CQA_RLHF/sft/dataset")
import argparse
from dataset import create_dataloaders
from accelerate_trainer import Trainer
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
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
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
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
        "--deepsped_config",
        type=str,
        default=None,
        help="File with DeepSpeed config file.",
    )

    args = parser.parse_args()

    config = {
        "seed": 42,
        'batch_size': 8,
        'max_legth': 512,
        "learning_rate": 2e-5,
        "max_steps": 195000,
        "eval_every": 500,
        "warmup_steps": 100,
        "wandb_kwargs": {"entity": "myashka", "job_type": "train", "group": "sft"},
        'TPU': False,
    }

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
        logging_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cpu=args.cpu,
        deepspeed_plugin=args.deepsped_config,
        downcast_bf16=True if config['TPU'] else False,
    )
    trainer = Trainer(
        max_steps=config["max_steps"],
        eval_every=config["eval_every"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=config["seed"],
        log_with=args.log_with,  # None, 'all', 'tensorboard', 'wandb', 'comet_ml'
        output_dir=args.output_dir,
        max_grad_norm=None,
        tracker_init_kwargs=config['wandb_kwargs'],
        cpu=args.cpu,
        resume_from_checkpoint=args.resume_from_checkpoint,
        accelerator=accelerator
    )

    model_name = config["model_name"]
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.pad_token_id = tokenizer.eos_token_id

    train_loader, val_loader = create_dataloaders(args.data_path,
                                                  tokenizer,
                                                  spltis=['train', 'val'],
                                                  batch_sizes=[config['batch_size'], config['batch_size']],
                                                  max_length=config['max_length'],
                                                  all_max_length=config['TPU'])

    trainer.train(model, tokenizer, train_loader, val_loader)

if __name__ == "main":
    main()