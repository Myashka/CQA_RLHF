# Supervised Fine-Tuning (sft)

## Requirements
This file lists the Python libraries that your project depends on. When you set up your environment, you can use this file to install all necessary dependencies in one command, using pip:

```
pip install -r requirements.txt
```

## Data

The project operates on a dataset in JSON format, which is comprised of question and answer pairs. This dataset is used both in training the model (Supervised Fine-Tuning) and evaluating it (Inference).

### Input Data

The input data should be in a JSON file with the following structure:

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

Each element in the "train" and "test" arrays represents a question-answer pair.

The **`prepare_train`** function in the provided Python script processes the training part of this dataset. For each question-answer pair, it tokenizes both the question and the answer, truncates them if required, and constructs an input sequence formatted as "Question: [question] \nAnswer:". It also assigns labels to the input sequence for the training process.

The **`prepare_inference`** function prepares the test part of the dataset for the evaluation stage. It tokenizes the question part and pads the sequence to the desired length.

In both cases, the functions generate "input_ids" and "attention_mask" for each entry, which are used by the GPT-Neo model for the training and inference stages.

Please ensure your input data file is located at the path specified in the respective configuration files (trainer_config.json for training and test_config.json for testing). The data paths are specified under the **`data_dir`** field in these files.

## Configuration

### Trainer_config.yaml

This configuration file sets up the parameters for the training process in the Supervised Fine-Tuning (sft) phase. Here's an explanation of each section:

**data**

- **`batch_size`**: The number of training examples utilized in one iteration.

- **`data_dir`**: The directory where the training data is located.

- **`max_length`**: The maximum length of the sequence for the model.

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

### Test_config.yaml

The testing configuration file is used to specify various parameters for testing the model. Here's what each section and parameter stands for:

**`cuda`**
This flag indicates whether to use a CUDA compatible device (GPU) for model inference.

**data**
This section contains the settings related to the testing data.

- **`data_dir`**: The directory where the testing data is located.
- **`max_prompt_length`**: The maximum length of the prompt for the model.
- **`padding`**: A flag to indicate if padding should be applied to the input sequences.
- **`padding_side`:** The side ("right" or "left") where padding should be applied.
- **`split`**: The dataset split to be used for testing, typically "test".
- **`truncate_promt`**: A flag to indicate if the prompt should be truncated to max_prompt_length.

**generate_params**
This section contains parameters related to the text generation process.

- **`do_sample`**: A flag to indicate if sampling should be used for text generation.
- **`max_new_tokens`**: The maximum number of tokens to generate.
- **`no_repeat_ngram_size`**: The size of the no-repeat n-gram. This parameter can be used to prevent the model from repeating the same phrase.
- **`top_k`**: The number of top choices to consider for each prediction.
- **`top_p`**: The cumulative probability cutoff for the top choices.

**`model_name`**: This is the identifier of the pretrained model from the transformers model hub.

**`seed`**
The seed for random number generators to achieve reproducible results.

**test_params**
This section contains parameters related to the testing process.

- **`do_compute_metrics`**: A flag to indicate if BLEU and ROUGE should be computed during testing.
- **`log_file`**: The file where the logs should be written.
- **`save_steps`**: The frequency of saving the generated text during testing.
- **`test_model_path`**: The path to the model to be tested.
use_cache: A flag to indicate if the model should cache the outputs of the transformers.

**wandb**

This section contains the settings related to Weights & Biases integration.

- **`api`**: The type of API for logging.
- **args**: Contains parameters for the Weights & Biases run.
    - **`group`**: The group of the run.
    - **`job_type`**: The type of job for the run.
    - **`name`**: The name of the run.
    - **`project_name`**: The name of the Weights & Biases project.

Please adjust these parameters as necessary for your specific use case.

## Run on Colab

**run.ipynb**

This is a Jupyter notebook file which can be used to run the training and testing scripts interactively from environments like Google Colaboratory. To use it, upload it to your Jupyter or Colab environment and follow the steps in the cells.