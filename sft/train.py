import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

import yaml
from yaml import CLoader
import click
from model import LitLM
from data_module import QADataModule
from pytorch_lightning.callbacks import ModelCheckpoint
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

    dm = QADataModule(config["model_name"], **config["data"])
    llm = LitLM(
        model_name=config["model_name"],
        batch_size=config["data"]["batch_size"],
        max_length=config["data"]["max_length"],
        **config["model_params"],
    )
    wandb_logger = WandbLogger(
        project=config["wandb"]["project_name"],
        log_model="all",
        **config["wandb"]["args"],
    )

    wandb_logger.watch(llm, log_graph=False)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        monitor=["trainer"]["checkpoint"]['log_obg'],
        mode=["trainer"]["checkpoint"]['mode'],
        filename="gpt-neo-sft-{epoch:02d}-{val_loss}",
    )
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
        llm,
        datamodule=dm,
        ckpt_path=config["trainer"]["ckpt_path"],
    )


if __name__ == "__main__":
    main()
