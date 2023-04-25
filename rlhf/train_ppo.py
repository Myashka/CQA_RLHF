import torch
from tqdm import tqdm
tqdm.pandas()

from transformers import AutoTokenizer

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed
from data.data_utils import build_dataset, collator
from reward_pipelines.regression_reward import Reward_pipeline
from training_utils.freeze import freeze_model

import yaml
from yaml import CLoader
import click



@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):

    with open(config_file, "r") as f:
        args_config = yaml.load(f, Loader=CLoader)

    config = PPOConfig(**args_config['ppo_config'])

    data_config = args_config['data_config']

    reward_config = args_config['reward_config']

    set_seed(config.seed)

    dataset = build_dataset(config, data_config, ['train'])[0]

    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = freeze_model(model, args_config['freeze_config'])

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator, num_shared_layers=None)

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    reward_pipe = Reward_pipeline(reward_config['model_name'], device)

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 512,
    }

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        response_tensors = []

        for query in query_tensors:
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze())
        batch["question_answer"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        rewards = reward_pipe(batch["question_answer"], reward_config['batch_size'])
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
    
    ppo_trainer.model.push_to_hub(args_config['hf_hub_name'])

if __name__ == "__main__":
    main()