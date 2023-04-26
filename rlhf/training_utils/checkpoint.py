from os.path import join as opj
import wandb

def save_checkpoint(model, run, epoch, reward, checkpoint_dir, name):
    checkpoint_name = f'{name}-_reward-{reward}_epoch-{epoch}'
    model.save_pretrained(opj(checkpoint_dir, checkpoint_name))

    artifact = wandb.Artifact(f'checkpoint_name', type='model', metadata={'mean_reward': reward, 'epoch': epoch})
    artifact.add_dir(opj(checkpoint_dir, checkpoint_name))
    run.log_artifact(artifact)