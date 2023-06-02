import gc
import os

import click
import nltk
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
import yaml
from models import sft_model
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import SacreBLEUScore
from torchmetrics.text.rouge import ROUGEScore
from tqdm.auto import tqdm
from yaml import CLoader

from data import QADataModule


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

    nltk.download("punkt")

    if config["cuda"] and torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = "cpu"

    pl.seed_everything(config["seed"])

    wandb.login(key=config["wandb"]["api"])

    if config['test_params']["test_model_path"] is not None:
        model = sft_model.LitLM.load_from_checkpoint(
            config['test_params']["test_model_path"]).to(device)
    else:
        model = sft_model.LitLM(
            model_name=config["model_name"],
            use_cache=config['test_params']["use_cache"],
        ).to(device)

    if config['test_params']["use_cache"]:
        model.model.config.use_cache = True

    dm = QADataModule(model_name=config["model_name"], **config["data"])
    dm.setup('test')
    test_dataset = dm.test_ds

    wandb_logger = WandbLogger(
        project=config["wandb"]["project_name"],
        log_model=True,
        **config["wandb"]["args"],
    )

    wandb_logger.watch(model, log_graph=False)

    columns = ["question", "original_answer", "generated_answer"]

    if config['test_params']["do_compute_metrics"]:
        rouge = ROUGEScore()
        bleu = SacreBLEUScore(1, lowercase=True)
        columns.append("bleu")
        columns.append("rouge1_fmeasure")
        columns.append("rouge2_fmeasure")
        columns.append("rougeL_fmeasure")

    model.eval()

    test_data = []
    step_processed = 0
    for sample in tqdm(test_dataset):
        test_sample = []
        gen_question_answer = model.generate(
            sample['input_ids'], sample['attention_mask'], device, **config["generate_params"]
        )
        promt_len = len(model.tokenizer.decode(sample['input_ids'], skip_special_tokens=True))
        
        gen_answer = str(gen_question_answer[promt_len:])
        

        test_sample.append(sample['Question'])
        test_sample.append(sample['Answer'])
        test_sample.append(gen_answer)

        if config['test_params']["do_compute_metrics"]:

            rouge_score = rouge(gen_answer, sample['Answer'])
            bleu_score = bleu(gen_answer, sample['Answer'])

            test_sample.append(bleu_score.item())

            test_sample.append(rouge_score["rouge1_fmeasure"].item())
            test_sample.append(rouge_score["rouge2_fmeasure"].item())
            test_sample.append(rouge_score["rougeL_fmeasure"].item())

        test_data.append(test_sample)
        assert test_data[-1] is not None, 'Something go wrong!'

        step_processed += 1
        if step_processed % config['test_params']['save_steps'] == 0:
            save_csv(test_data, columns, config['test_params']['log_file'])
            test_data = []
            gc.collect()

    save_csv(test_data, columns, config['test_params']['log_file'])
    wandb.finish()
    # log the Table
    # wandb_logger.log_table(key=config['wandb']["table_name"], columns=columns, data=test_data)


if __name__ == "__main__":
    main()
