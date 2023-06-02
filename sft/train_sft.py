import os

import click
import pytorch_lightning as pl
import wandb
import yaml
from models import sft_model
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from yaml import CLoader

from data import QADataModule


@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):

    wandb.require("service")

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)

    pl.seed_everything(config["seed"])

    wandb.login(key=config["wandb"]["api"])

    dm = QADataModule(model_name=config["model_name"], **config["data"])
    if config["trainer"]["ckpt_path"]:
        llm = sft_model.LitLM.load_from_checkpoint(config["trainer"]["ckpt_path"], **config["model_params"])
    else:
        llm = sft_model.LitLM(
        model_name=config["model_name"],
        **config["model_params"])
        
    wandb_logger = WandbLogger(
        project=config["wandb"]["project_name"],
        log_model="all",
        **config["wandb"]["args"],
    )

    wandb_logger.watch(llm, log_graph=False)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        monitor=config["trainer"]["checkpoint"]['log_obg'],
        mode=config["trainer"]["checkpoint"]['mode'],
    )
    # checkpoint_callback = ModelCheckpoint(
    #     every_n_train_steps=config["trainer"]["checkpoint"]["every_n_train_steps"],
    #     filename="gpt-neo-sft-{epoch:02d}-{global_step}",
    #     dirpath=config["trainer"]["checkpoint"]["dirpath"],
    # )

    if config['trainer']['params']['accelerator'] == 'gpu':
        trainer = pl.Trainer(
            logger=wandb_logger,
            default_root_dir=os.getcwd(),
            callbacks=[checkpoint_callback, lr_monitor],
            # strategy=DeepSpeedStrategy(logging_batch_size_per_gpu=config["data"]["batch_size"]),
            **config["trainer"]["params"],
        )
    else:
        trainer = pl.Trainer(
            logger=wandb_logger,
            default_root_dir=os.getcwd(),
            callbacks=[checkpoint_callback, lr_monitor],
            **config["trainer"]["params"],
        )

    trainer.fit(
        llm,
        datamodule=dm,
        # ckpt_path=config["trainer"]["ckpt_path"],
    )
    wandb.finish()


if __name__ == "__main__":
    main()
