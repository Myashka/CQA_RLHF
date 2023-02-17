# Description

The sft directory is the most important part of this repository as it contains the code for fine-tuning the GPT-Neo model on the CQA task. The directory contains several Python scripts that implement the fine-tuning process using the Hugging Face Transformers library.

The main script is train.py, which takes the pre-trained GPT-Neo model and fine-tunes it on the CQA data. The script allows users to configure various hyperparameters such as learning rate, batch size, and number of training epochs via config.yaml