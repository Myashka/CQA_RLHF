import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

import yaml
from yaml import CLoader
import click
from model import GPTneo_Regressor
from data_module import QA_Reward_DataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor


@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):

    wandb.require("service")

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)

    pl.seed_everything(config["seed"])

    wandb.login(key=config["wandb"]["api"])

    dm = QA_Reward_DataModule(config["model_name"], **config["data"])
    regressor = GPTneo_Regressor(
        model_name=config["model_name"],
        batch_size=config["data"]["batch_size"],
        **config["model_params"],
    )
    wandb_logger = WandbLogger(
        project=config["wandb"]["project_name"],
        log_model="all",
        **config["wandb"]["args"],
    )

    wandb_logger.watch(regressor, log_graph=False)
    checkpoint_callback = ModelCheckpoint(
        monitor=config["trainer"]["checkpoint"]['log_obg'],
        mode=config["trainer"]["checkpoint"]['mode'],
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    # earlystopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    # checkpoint_callback = ModelCheckpoint(
    #     every_n_train_steps=config["trainer"]["checkpoint"]["every_n_train_steps"],
    #     filename="gpt-neo-sft-{epoch:02d}-{global_step}",
    #     dirpath=config["trainer"]["checkpoint"]["dirpath"],
    # )

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
