# Regression reward model training

## Requirements
This file lists the Python libraries that your project depends on. When you set up your environment, you can use this file to install all necessary dependencies in one command, using pip:

```
pip install -r requirements.txt
```

## Data

The project operates on a dataset in JSON format, which is comprised  of question and answer pairs and an associated score.

### Input Data

```
{
    "train": [
        {
            "Question": "Example question?",
            "Answer": "Example answer.",
            "Score": 5
        },
        ...
    ],
    "test": [
        {
            "Question": "Example question?",
            "Answer": "Example answer.",
            "Score": 5
        },
        ...
    ]
}

```

Each element in the "train" and "test" arrays represents a question and two-answer set.

The **`prepare_train`** function in the provided Python script processes this dataset for the training process. For each question and two-answer set, it tokenizes the question, the accepted answer, and the additional answer, truncating them if required, and constructs two input sequences. Each sequence is formatted as "Question: [question] \nAnswer:" - one for the accepted answer and one for the other answer.

The function generates "input_ids" and "attention_mask" for each entry. There are two sets of these for each question: one for the accepted answer (input_ids_j, attention_mask_j)

## Configuration

### Trainer_config.yaml

This configuration file sets up the parameters for the training process in the reward model training phase. Here's an explanation of each section:

**data**

- **`batch_size`**: The number of training examples utilized in one iteration.

- **`data_dir`**: The directory where the training data is located.

- **`max_length`**: The maximum length of the sequence for the model.

- **`max_prompt_length`**: The maximum length of prompt.

- **`on_tpu`**: A flag to indicate if training is on a TPU.

- **`truncate_promt`** : A flag to indicate if the prompt should be truncated to max_length.

**model_name**
This is the identifier of the pretrained model from the transformers model hub.

**model_params**
This section contains the hyperparameters related to the model.

- **`adam_betas`**: The beta parameters for the Adam optimizer.
- **`do_compute_metrics`**: A flag to indicate if metrics should be computed during training.
- **`learning_rate`**: The learning rate for the optimizer.
- **`use_cache`**: A flag to indicate if the model should cache the outputs of the transformers.
warmup_steps_per_cent: The percentage of total steps used for the learning rate warmup.
- **`weight_decay`**: The L2 penalty (regularization term) parameter.
- **`do_freeze`**: A flag to indicate if parts of the model should be frozen during training.
- **`freeze_emb`**: A flag to indicate if the embeddings should be frozen.
- **`freeze_ln`**: A flag to indicate if the layer normalization should be frozen.
- **`freeze_attn`**: A flag to indicate if the attention layers should be frozen.
- **`freeze_ff`**: A flag to indicate if the feed-forward layers should be frozen.
- **`freeze_other`**: A flag to indicate if other layers should be frozen.
- **`layers_not_to_freeze`**: A list of layers that should not be frozen during training.

**seed**
The seed for random number generators to achieve reproducible results.

**trainer**

This section contains the settings related to the training process.

- **`checkpoint`**: Contains settings related to model checkpoints.
    - **`dirpath`**: The directory where checkpoints are saved.
    - **`every_n_train_steps`**: The frequency of saving checkpoints in terms of training steps.
    - **`log_obg`**: The variable used to select the best checkpoint.
    - **`mode`**: Determines whether the best checkpoint is the one with maximum or minimum log_obg.

- **`ckpt_path`**: The path to a specific checkpoint to load.
- **params**: Contains various parameters for the training process.
    - **`accelerator`**: The type of hardware accelerator to be used during training.
    - **`accumulate_grad_batches`**: The number of batches to process before performing a backward/update pass.
    - **`gradient_clip_val`**: The maximum value for gradient clipping.
    - **`log_every_n_steps`**: The frequency of logging in terms of training steps.
    - **`max_epochs`**: The maximum number of epochs for training.
    - **`num_sanity_val_steps`**: The number of validation samples used for the sanity check.
    - **`overfit_batches`**: Defines how much of training data should be used for overfitting.
    - **`precision`**: The precision of the gradients.
    - **`val_check_interval`**: The frequency of validation in terms of training steps.

**wandb**

This section contains the settings related to Weights & Biases integration.

- **`api`**: The type of API for logging.
- **args**: Contains parameters for the Weights & Biases run.
    - **`group`**: The group of the run.
    - **`job_type`**: The type of job for the run.
    - **`name`**: The name of the run.
    - **`project_name`**: The name of the Weights & Biases project.

Please modify these fields as required for your specific needs.