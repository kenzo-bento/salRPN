#!/usr/bin/env python3

import cProfile
import pstats
from utils.run_experiment import run_experiment
import torch

device = torch.device('cuda')

print("DEVICE: ", device)

def main_training_function():
    # Run one experiment with saliency-informed hard dropout
    train_stats = run_experiment(dataset='coco',
                    sal_model='ProtoObject',
                    sal_type = 'mask',
                    sal_method = 'soft_sal',
                    model_name='resnet',
                    wandb_name = 'SoftSal',
                    run_outputs_path='./run_outputs/',
                    optimizer_name='SGD', # 'SGD', 'Adam'
                    scheduler_name= None,  # 'ExponentialLR', 'StepLR', or None
                    num_workers=0,
                    epochs=200,
                    batch_size=1,
                    nTrain=2,
                    nTest=2,
                    learning_rate=1e-3,
                    report_wandb=True,
                    save_model=True,
                    save_stats=True,
                    save_f_maps=False,
                    seed_val=42,
                    k = 0.1)
    pass

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    main_training_function()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime') # Sort by total time spent in function
    stats.print_stats(15) # Print the top 15 time-consuming functions
