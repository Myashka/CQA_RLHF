import os
os.chdir('/content/trlx')
from typing import List

import torch
from datasets import load_dataset
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer, GPTNeoForSequenceClassification

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

REWARD_CHECKPOINT_PATH = "Myashka/125M_GPTneo_reward_gen"
SFT_MODEL_PATH = "Myashka/125M_GPTneo_sft_tuned"
DATA_PATH = "/kaggle/input/python-qa-api-usage/1.0-data-div-ans-sep-api-usage.json"

config = TRLConfig(
    train=TrainConfig(
        seq_length=512,
        epochs=50,
        total_steps=100000,
        batch_size=4,
        checkpoint_interval=10000,
        eval_interval=200,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
    ),
    model=ModelConfig(
        model_path=SFT_MODEL_PATH,
        num_layers_unfrozen=8,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path=SFT_MODEL_PATH,
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 5.0e-6,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 100000,
            "eta_min": 5.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=128,
        chunk_size=16,
        ppo_epochs=4,
        init_kl_coef=0.1,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.2,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 128,
            "min_new_tokens": 64,
        },
        log_with='wandb',
    ),
)


if __name__ == "__main__":
    # Load the pre-trained reward model
    wandb.require("service")

    rw_tokenizer = AutoTokenizer.from_pretrained(REWARD_CHECKPOINT_PATH)
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    rw_model = GPTNeoForSequenceClassification.from_pretrained(REWARD_CHECKPOINT_PATH)
    rw_model.pad_token_id = rw_tokenizer.eos_token_id
    rw_model.config.end_token_id = rw_tokenizer.eos_token_id
    rw_model.config.pad_token_id = rw_model.config.eos_token_id
    
    # rw_model.half()
    rw_model.eval()
    rw_device = torch.device("cuda:{}".format(1))  # set reward model device
    rw_model.to(rw_device)

    def get_scores(samples: List[str]):
        scores_list = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i : i + batch_size]
            # sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
            encodings_dict = rw_tokenizer(
                sub_samples,
                truncation=True,
                max_length=config.train.seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(rw_device)
            attn_masks = encodings_dict["attention_mask"].to(rw_device)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            scores_list.append(sub_scores.logits)
        scores = torch.cat(scores_list, dim=0)
        return scores

    def get_prompt_dataset(prompts, max_length):

        formatted_prompts = []
        for i in tqdm(range(len(prompts))):
            tmp = tokenizer.decode(
                tokenizer(
                    prompts[i],
                    truncation=True,
                    max_length=max_length - 10,
                    add_special_tokens=False,
                )["input_ids"],
                skip_special_tokens=True,
            ).strip()
            tmp = 'Question: ' + tmp + "\nAnswer:"
            tmp = tokenizer.decode(
                tokenizer(tmp, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"],
                skip_special_tokens=True,
            ).strip()
            formatted_prompts.append(tmp)
        return formatted_prompts

    def reward_fn(samples: List[str], **kwargs):
        original_samples = [text.split("\nAnswer:")[0] + "\nAnswer: " for text in samples]
        original_samples = [text + question_answer_dict[text.strip()] for text in original_samples]
        original_scores = get_scores(original_samples)
        scores = get_scores(samples)
        norms_scores = scores - original_scores
        return norms_scores

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "rigth"
    max_length_input = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    train_dataset = load_dataset("json", data_files=DATA_PATH, field='train')['train']
    val_dataset = load_dataset("json", data_files=DATA_PATH, field='val')['train']

    # Store data into prompt and label pairs
    train_set = [(sample["Question"], sample["Answer"]) for sample in train_dataset]
    val_set = [(sample["Question"], sample["Answer"]) for sample in val_dataset]

    # Split contents into summaries and labels
    train_questions, train_answers = zip(*train_set)
    val_questions, val_answers = zip(*val_set)

    # Get the OpenAI summaries
    question_answer_dict = {}
    train_prompts = get_prompt_dataset(train_questions, max_length_input)
    for i in range(len(train_prompts)):
        question_answer_dict[train_prompts[i]] = train_answers[i]
    val_prompts = get_prompt_dataset(val_questions, max_length_input)
    for i in range(len(val_prompts)):
        question_answer_dict[val_prompts[i]] = val_answers[i]

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts[0:1000],  # sampling 1000 validation prompts for evaluation speed in training
        config=config,
    )