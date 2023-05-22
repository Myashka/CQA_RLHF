# RLHF model training

## Requirements
This file lists the Python libraries that your project depends on. When you set up your environment, you can use this file to install all necessary dependencies in one command, using pip:

```
pip install -r requirements.txt
```

## Data

The project operates on a dataset in JSON format, which is comprised of question and answer pairs.

### Input Data

```
{
    "train": [
        {
            "Question": "Example question?",
            "Answer": "Example answer."
        },
        ...
    ],
    "test": [
        {
            "Question": "Example question?",
            "Answer": "Example answer."
        },
        ...
    ]
}
```

## Configuration

### Trainer_config.yaml

The RLHF trainer configuration file is used to manage and control the different aspects of the training process in Reinforcement Learning from Human Feedback (RLHF). The following is a description of each of the configurations:

**data_config**: Configuration related to data preprocessing and loading.

- **`data_file_path`**: The path to the input data file for RLHF.

- **`max_length_promt`**: Maximum length of the prompt in tokens.

- **`padding`**: Whether to pad the sequences to the maximum length.

- **`truncate_promt`**: Whether to truncate the prompts to the maximum length.

**freeze_config**: Specifies which parts of the model to freeze during training.

- **`freeze_emb`**: A flag to indicate if the embeddings should be frozen.
- **`freeze_ln`**: A flag to indicate if the layer normalization should be frozen.
- **`freeze_attn`**: A flag to indicate if the attention layers should be frozen.
- **`freeze_ff`**: A flag to indicate if the feed-forward layers should be frozen.
- **`freeze_other`**: A flag to indicate if other layers should be frozen.
- **`layers_not_to_freeze`**: A list of layers that should not be frozen during training.

**generation_config**: Settings for the text generation process.

- **`do_sample`**: Whether to use sampling in the generation process.
- **`max_new_tokens`**: The maximum number of tokens to generate.
- **`min_length`**: The minimum length of the generated text.
- **`top_k`**: The number of highest probability vocabulary tokens to keep for top-k-filtering.
- **`top_p`**: The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling.

**`global_epoches`**: The total number of epochs to train the model.

**`hf_hub_name`**: The name of the model on the Hugging Face model hub.

**ppo_config**: Configuration for the Proximal Policy Optimization (PPO) algorithm.

- **`batch_size`**: The batch size for PPO.
**`forward_batch_size`**: The forward batch size used in PPO.
**`gradient_accumulation_steps`**: The number of steps for gradient accumulation.
**`learning_rate`**: The learning rate for the optimizer.
**`log_with`**: The logging utility to use (in this case, Weights and Biases).
**`max_grad_norm`**: The maximum gradient norm.
**`model_name`**: The name of the model being fine-tuned.
**`optimize_cuda_cache`**: Whether to optimize CUDA cache for better performance.
**`ppo_epochs`**: The number of epochs for PPO.
**`remove_unused_columns`**: Whether to remove unused columns in the dataset.
**`seed`**: The random seed for reproducibility.
**`steps`**: The number of steps for PPO.
**`tracker_kwargs`**: Configuration for the experiment tracker.
**`tracker_project_name`**: The name of the project in the experiment tracker.

**reward_config**: Configuration for the reward model.

- **`batch_size`**: The batch size for the reward model.
- **`reward_model_name`**: The name of the reward model.

**save_config**: Settings related to model saving.

- **`checkpoint_dir`**: The directory to save the checkpoints.
- **`save_interval`**: The interval (in terms of steps) at which to save the model.

Please adjust these configurations based on your requirements and resources for the best results.