import os
from os.path import join as opj

import torch
import wandb


def save_checkpoint(model, run, global_epo, epoch, reward, checkpoint_dir, name, run_name):
    checkpoint_name = f'{name}-glob_epo_{global_epo}-reward_{round(reward.item(), 2)}-epoch_{epoch}'
    os.makedirs(opj(checkpoint_dir, checkpoint_name), exist_ok=True)
    torch.save(model.state_dict(), opj(checkpoint_dir, checkpoint_name)+'/pytorch_model.pt')

    artifact = wandb.Artifact(f"{run_name}-{name}", type='model', metadata={'mean_reward': reward, 'epoch': epoch})
    artifact.add_dir(opj(checkpoint_dir, checkpoint_name))
    run.log_artifact(artifact)