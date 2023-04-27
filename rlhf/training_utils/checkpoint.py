from os.path import join as opj
import wandb

def save_checkpoint(model, run, global_epo, epoch, reward, checkpoint_dir, name):
    checkpoint_name = f'{name}-glob_epo_{global_epo}-reward_{round(reward.item(), 2)}-epoch_{epoch}'
    model.save_pretrained(opj(checkpoint_dir, checkpoint_name))

    artifact = wandb.Artifact(name, type='model', metadata={'mean_reward': reward, 'epoch': epoch})
    artifact.add_dir(opj(checkpoint_dir, checkpoint_name))
    run.log_artifact(artifact)