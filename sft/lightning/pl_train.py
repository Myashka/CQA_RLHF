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


@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)

    pl.seed_everything(config["seed"])

    wandb.login(key=config["wandb"]["api"])
    wandb_logger = WandbLogger(
        project=config["wandb"]["project_name"],
        log_model=True,
        **config["wandb"]["args"]
    )

    dm = QADataModule(
        config["model_name"],
        config["data"]["path_to_data"],
        config["data"]["max_length"],
        config["data"]["batch_size"],
    )
    llm = LitLM(
        config["model_name"],
        config["model_params"]["lr"],
        config["model_params"]["do_freeze"],
        config["model_params"]["use_cache"],
        config["model_params"]["warmup_steps"],
        config["model_params"]["adam_betas"],
        config["model_params"]["weight_decay"],
    )

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=config["trainer"]["checkpoint"]["every_n_train_steps"],
        filename="gpt-neo-sft-{epoch:02d}-{global_step}",
        dirpath=config["trainer"]["checkpoint"]["dirpath"],
    )

    trainer = pl.Trainer(
        auto_scale_batch_size=config["trainer"]["auto_scale_batch_size"],
        accelerator=config["trainer"]["accelerator"],
        max_epochs=config["trainer"]["max_epochs"],
        logger=wandb_logger,
        accumulate_grad_batches=config["trainer"]["accumulate_grad_batches"],
        gradient_clip_val=config["trainer"]["gradient_clip_val"],
        default_root_dir=os.getcwd(),
        callbacks=[checkpoint_callback],
        val_check_interval=config["trainer"]["val_check_interval"],
        precision=config["trainer"]["precision"],
    )
    trainer.fit(
        llm,
        datamodule=dm,
        ckpt_path=config["trainer"]["ckpt_path"],
    )


if __name__ == "__main__":
    main()
