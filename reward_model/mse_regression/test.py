import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

import yaml
from yaml import CLoader
import click
from model import GPTneo_Regressor
from data_module import QA_Reward_DataModule
import torch
from tqdm.auto import tqdm


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

    if config["test_model_path"] is not None:
        model = GPTneo_Regressor.load_from_checkpoint(
            config["test_model_path"]).to(device)
        if config["model_params"]["use_cache"]:
            model.config.use_cache = True
    else:
        model = GPTneo_Regressor(
            model_name=config["model_name"],
            **config["model_params"],
            **config['test_params']
        ).to(device)

    test_dataset = QA_Reward_DataModule(
        config["data"]["data_dir"], model.tokenizer, split="test")

    wandb_logger = WandbLogger(
        project=config["wandb"]["project_name"],
        log_model=True,
        **config["wandb"]["args"],
    )

    wandb_logger.watch(model, log_graph=False)

    columns = ["qa_pair", "target_score", "pred_score", 'mse']

    model.eval()

    test_data = []
    for sample in tqdm(test_dataset):
        test_sample = []
        model_output = model(**sample).to("cpu")

        test_sample.append(model.tokenizer.decode(sample['input_ids']))
        test_sample.append(sample['labels'])
        test_sample.append(model_output.logits)
        test_sample.append(model_output.loss)

        test_data.append(test_sample)

    # log the Table
    wandb_logger.log_table(
        key=config["table_name"], columns=columns, data=test_data)


if __name__ == "__main__":
    main()
