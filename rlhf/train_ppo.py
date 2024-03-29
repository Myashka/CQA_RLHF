import torch
import wandb
from tqdm import tqdm

tqdm.pandas()

import gc

import click
import yaml
from reward_pipelines import Reward_pipeline
from training_utils import freeze_model, save_checkpoint
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from yaml import CLoader

from rlhf.data import build_dataset, collator


@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):
    with open(config_file, "r") as f:
        args_config = yaml.load(f, Loader=CLoader)

    config = PPOConfig(**args_config["ppo_config"])

    data_config = args_config["data_config"]
    freeze_config = args_config["freeze_config"]
    reward_config = args_config["reward_config"]
    save_config = args_config["save_config"]

    generation_kwargs = args_config["generation_config"]

    config.data_config = data_config
    config.freeze_config = freeze_config
    config.save_config = save_config
    config.generation_kwargs = config
    config.reward_config = reward_config

    set_seed(config.seed)

    dataset = build_dataset(config, data_config, ["train"])[0]

    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    model = freeze_model(model, freeze_config)

    tokenizer.pad_token = tokenizer.eos_token

    print("Trainer start")
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model,
        tokenizer,
        dataset=dataset,
        data_collator=collator,
        num_shared_layers=None,
    )
    print("Trainer done")

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    reward_pipe = Reward_pipeline(
        reward_config["reward_model_name"], ppo_trainer.accelerator
    )

    wandb_tracker = ppo_trainer.accelerator.get_tracker("wandb", unwrap=True)

    generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
    best_reward = -100

    global_epoches = args_config["global_epoches"]

    for global_epo in tqdm(range(global_epoches)):
        for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            query_tensors = batch["input_ids"]
            response_tensors = []

            for query in query_tensors:
                response = ppo_trainer.generate(query, **generation_kwargs)
                response_tensors.append(response.squeeze())
            # batch["question_answer"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            # batch["query"] = [tokenizer.decode(query_idx, skip_special_tokens=True) for query_idx in batch["input_ids"]]
            batch["response"] = [
                tokenizer.decode(
                    r.squeeze()[len(query_idx) :], skip_special_tokens=True
                )
                for r, query_idx in zip(response_tensors, batch["input_ids"])
            ]

            #### Compute sentiment score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            rewards = reward_pipe(texts, reward_config["batch_size"])

            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

            mean_reward = torch.mean(torch.tensor(rewards))

            del batch
            del rewards
            gc.collect()
            torch.cuda.empty_cache()

            if (epoch + 1) % save_config["save_interval"] == 0:
                ppo_trainer.accelerator.wait_for_everyone()
                if ppo_trainer.accelerator.is_main_process:
                    unwrapped_model = ppo_trainer.accelerator.unwrap_model(
                        ppo_trainer.model
                    )
                    save_checkpoint(
                        unwrapped_model,
                        wandb_tracker,
                        global_epo,
                        epoch,
                        mean_reward,
                        save_config["checkpoint_dir"],
                        "ppo_checkpoint",
                        config.tracker_kwargs["name"],
                    )

            if mean_reward > best_reward:
                ppo_trainer.accelerator.wait_for_everyone()
                if ppo_trainer.accelerator.is_main_process:
                    unwrapped_model = ppo_trainer.accelerator.unwrap_model(
                        ppo_trainer.model
                    )
                    save_checkpoint(
                        unwrapped_model,
                        wandb_tracker,
                        global_epo,
                        epoch,
                        mean_reward,
                        save_config["checkpoint_dir"],
                        "max_reward_ppo",
                        config.tracker_kwargs["name"],
                    )

                best_reward = mean_reward

        ppo_trainer.accelerator.wait_for_everyone()
        if ppo_trainer.accelerator.is_main_process:
            unwrapped_model = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
            save_checkpoint(
                unwrapped_model,
                wandb_tracker,
                global_epo,
                epoch,
                mean_reward,
                save_config["checkpoint_dir"],
                "last_checkpoint",
                config.tracker_kwargs["name"],
            )

    ppo_trainer.accelerator.wait_for_everyone()
    if ppo_trainer.accelerator.is_main_process:
        ppo_trainer.model.push_to_hub(args_config["hf_hub_name"])


if __name__ == "__main__":
    main()
