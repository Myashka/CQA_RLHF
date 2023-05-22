import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

import yaml
from yaml import CLoader
import click
from models import GPTneo_Regressor
from data import QA_Reward_DataModule
import os
import torch
from tqdm.auto import tqdm
import gc
import pandas as pd


def save_csv(data, columns, file_path):
    df = pd.DataFrame(data, columns=columns)
    if os.path.exists(file_path):
        mode = 'a'
        header = False
    else:
        mode = 'w'
        header = True
    df.to_csv(file_path, mode=mode, header=header)


@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)

    if config["cuda"] and torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = "cpu"

    pl.seed_everything(config["seed"])

    wandb.login(key=config["wandb"]["api"])

    if config['test_params']["test_model_path"] is not None:
        model = GPTneo_Regressor.load_from_checkpoint(
            config['test_params']["test_model_path"]).to(device)
    else:
        model = GPTneo_Regressor(
            model_name=config["model_name"],
            **config["model_params"],
            use_cache=config['test_params']["use_cache"],
        ).to(device)

    if config["test_params"]["use_cache"]:
        model.model.config.use_cache = True

    dm = QA_Reward_DataModule(
        model_name=config["model_name"], **config['data'])

    dm.setup('test')
    test_dataset = dm.test_ds

    wandb_logger = WandbLogger(
        project=config["wandb"]["project_name"],
        log_model=True,
        **config["wandb"]["args"],
    )

    wandb_logger.watch(model, log_graph=False)

    columns = ["question", 'answer', "target_score", "pred_score", 'mse']

    model.model.eval()

    test_data = []
    step_processed = 0
    for sample in tqdm(test_dataset):
        model_input = {"input_ids": sample['input_ids'].to(device),
                       "attention_mask": sample['attention_mask'].to(device),
                       'labels': sample['Score'].to(device)}
        test_sample = []
        model_output = model.model(**model_input)

        test_sample.append(sample['Question'])
        test_sample.append(sample['Answer'])
        test_sample.append(sample['Score'])
        test_sample.append(model_output.logits.cpu().detach().numpy()[0][0])
        test_sample.append(model_output.loss.item())

        gc.collect()

        test_data.append(test_sample)

        step_processed += 1
        if step_processed % config['test_params']['save_steps'] == 0:
            save_csv(test_data, columns, config['test_params']['log_file'])
            test_data = []

    save_csv(test_data, columns, config['test_params']['log_file'])
    # log the Table
    # wandb_logger.log_table(
    #     key=config['wandb']["table_name"], columns=columns, data=test_data)

    wandb.finish()


if __name__ == "__main__":
    main()
