# CQA_RLHF

## Descpription

This project structure describes an approach to fine-tuning GPT Neo model via a three-staged process that incorporates Supervised Fine-Tuning (SFT), Reward Model Training, and Reinforcement Learning from Human Feedback (RLHF). The repository consists of several folders, each representing a particular stage or functionality of the model. The structure is designed to allow easy training and testing with the ability to modify different configurations and use different datasets.

The project supports multi-GPU training, allowing you to leverage multiple graphics cards to significantly speed up the model training process using PyTorch-Lightning framework.

The project is structured as follows:
- **sft**: the SFT model, with its configuration files, data, and models.
- **reward_model**: contains the reward model with two different types of models - classification and regression.
- **rlhf**: the RLHF model, with its configuration files, data, reward pipelines, and training utilities.

## Usage
### Supervised Fine-Tuning (sft)

The **`sft`** folder contains the materials for the Supervised Fine-Tuning stage. You can modify the model's configurations in the 'configs' folder, include your datasets in the 'data' folder, and store or retrieve trained models in the 'models' folder.

Training awailable via next commands:
```
cd sft
python train_sft.py --config_file \configs\trainer_config.yaml
```

Testing sft and rlhf model available via next commands:
```
cd sft
python test_sft.py --config_file \configs\test_config.yaml
```

### Reward Model Training (reward_model)

This folder holds the reward model that is either based on contrast classification or sumple regression according to scaled scores. Both these models have their respective configuration, data, and model folders.

For train regression model:
```
cd reward_model/regression
python train_reward.py --config_file \configs\trainer_config.yaml
```
For test regression model:
```
cd reward_model/regression
python test_reward.py --config_file \configs\test_config.yaml
```
For classification model:
```
cd reward_model/classification
python train_reward.py --config_file \configs\trainer_config.yaml
```

### Reinforcement Learning from Human Feedback (rlhf)

The **rlhf** folder contains the RLHF model. Here, you can modify the RLHF model's configurations in the 'configs' folder, store your datasets in the 'data' folder, build or modify reward pipelines in the 'reward_pipelines' folder, and access the training utilities in the 'training_utils' folder.

```
cd rlhf
python train_ppo.py --config configs/trainer_config.yaml
```

## Requirements
The project requires Python 3.8+, as well as several Python libraries. For a full list of these libraries and their respective versions, please refer to the 'requirements.txt' files at each stage directory.

## Acknowledgements
Appreciation to the original authors and contributors of the GPT Neo model. This work wouldn't have been possible without their significant contributions to the field of Machine Learning.