import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

import yaml
from yaml import CLoader
import click
from model import LitLM
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import SacreBLEUScore
import nltk
from dataset import prepare_datasets
import torch
from tqdm.auto import tqdm
import pandas as pd
import os
import gc


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

    if config["test_model_path"] is not None:
        model = LitLM.load_from_checkpoint(
            config["test_model_path"]).to(device)
    else:
        model = LitLM(
            model_name=config["model_name"],
            **config["model_params"],
            **config['test_params']
        ).to(device)

    if config["model_params"]["use_cache"]:
        model.model.config.use_cache = True

    test_dataset = prepare_datasets(
        config["data"]["data_dir"], model.tokenizer, splits=["test"], train=False
    )[0]

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
        if config['test_params']["do_compute_bertscore"]:
            bertscore = BERTScore(lang="en")
            columns.append("bert_f1")
            columns.append("bert_precision")
            columns.append("bert_recall")

    model.eval()

    test_data = []
    step_processed = 0
    for question_promt, answer in tqdm(test_dataset):
        test_sample = []
        gen_question_answer = model.generate(
            question_promt, device, **config["generate_params"]
        )
        gen_answer = str(gen_question_answer[len(question_promt):])

        test_sample.append(question_promt)
        test_sample.append(answer)
        test_sample.append(gen_answer)

        if config['test_params']["do_compute_metrics"]:

            rouge_score = rouge(gen_answer, answer)
            bleu_score = bleu(gen_answer, answer)

            test_sample.append(bleu_score.item())

            test_sample.append(rouge_score["rouge1_fmeasure"].item())
            test_sample.append(rouge_score["rouge2_fmeasure"].item())
            test_sample.append(rouge_score["rougeL_fmeasure"].item())

            if config['test_params']["do_compute_bertscore"]:
                bert_score = bertscore(gen_answer, answer)
                test_sample.append(bert_score["f1"])
                test_sample.append(bert_score["precision"])
                test_sample.append(bert_score["recall"])

        test_data.append(test_sample)
        assert test_data[-1] is not None, 'Something go wrong!'

        step_processed += 1
        if step_processed % config['save_steps'] == 0:
            save_csv(test_data, columns, config['log_file'])
            test_data = []
            gc.collect()

    save_csv(test_data, columns, config['log_file'])
    wandb.finish()
    # log the Table
    # wandb_logger.log_table(key=config['wandb']["table_name"], columns=columns, data=test_data)


if __name__ == "__main__":
    main()
