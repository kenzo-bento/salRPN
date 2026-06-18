import torch.nn as nn
import wandb

def perform_run(trainerObj, base_model: nn.Module, es_params, wandb_params, report_wandb, wandb_name):
    train_stats = {}
    if report_wandb:
        
        wandb_name = wandb_name
            
        # initialize wandb run with the current prune iteration and parameters
        wandb.init(
            project="project-placeholder",
            entity = "entity-placeholder",
            name=wandb_name,
            config={
                "learning_rate": wandb_params['learning_rate'],
                "optimizer": wandb_params['optimizer'],
                "scheduler": wandb_params['scheduler'],
                "model": wandb_params['model'],
                "sal_model": wandb_params['sal_model'],
                "dataset": wandb_params['dataset'],
                "epochs": wandb_params['epochs'],
                "batch_size": wandb_params['batch_size'],
                "nTrain": wandb_params['nTrain'],
                "nTest": wandb_params['nTest'],
                "min_delta": wandb_params['min_delta'],
                "patience": wandb_params['patience'],
                }
            )

        # train until early stopping criteria is met
        train_stats = trainerObj.train_es(es_params=es_params)

        if report_wandb:
            wandb.finish()

    return train_stats
