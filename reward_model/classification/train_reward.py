import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

import yaml
from yaml import CLoader
import click
from model import GPTneo_Regressor
from data import QA_Reward_DataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor


@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):

    wandb.require("service")

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)

    pl.seed_everything(config["seed"])

    wandb.login(key=config["wandb"]["api"])

    dm = QA_Reward_DataModule(model_name=config["model_name"], **config["data"])
    regressor = GPTneo_Regressor(
        model_name=config["model_name"],
        **config["model_params"],
    )
    wandb_logger = WandbLogger(
        project=config["wandb"]["project_name"],
        log_model=True,
        **config["wandb"]["args"],
    )

    wandb_logger.watch(regressor, log_graph=False)
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        **config["trainer"]["checkpoint"]
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        logger=wandb_logger,
        default_root_dir=os.getcwd(),
        callbacks=[checkpoint_callback, lr_monitor],
        **config["trainer"]["params"],
    )

    trainer.fit(
        regressor,
        datamodule=dm,
        ckpt_path=config["trainer"]["ckpt_path"],
    )

    wandb.finish()


if __name__ == "__main__":
    main()
