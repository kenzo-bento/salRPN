#!/usr/bin/env python3

import argparse
from utils.run_experiment import run_experiment
import torch
import multiprocessing

def main():
    device = torch.device('cuda')
    print(f"Using device: {device}")

    # Define dropout rates for the experiment
    # The dropout rates are logarithmically spaced between 0.01 and 1.0, with an additional rate of 0.3
    # The last dropout rate is set to 1.0
    # This is done to ensure that the dropout rates are not too high, which can lead to poor performance
    # The dropout rates are used to control the amount of dropout applied during training

    parser = argparse.ArgumentParser(description='Run a training experiment.')
    parser.add_argument('--dataset', type=str, default='coco', help='Dataset to use')
    parser.add_argument('--sal_model', type=str, default='ProtoObject', help='Saliency model')
    parser.add_argument('--sal_type', type=str, default=None, help='Way of using saliency')
    parser.add_argument('--sal_method', type=str, default='maxima', help='Way of filtering')
    parser.add_argument('--model_name', type=str, default='resnet', help='Model architecture')
    parser.add_argument('--wandb_name', type=str, default='Maxima', help='Logging name')
    parser.add_argument('--run_outputs_path', type=str, default='./run_outputs/', help='Path to save outputs')
    parser.add_argument('--optimizer_name', type=str, default='SGD', help='Optimizer to use')
    parser.add_argument('--scheduler_name', type=str, default=None, help='Learning rate scheduler')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--nTrain', type=int, default=118000, help='Number of training samples')
    parser.add_argument('--nTest', type=int, default=5000, help='Number of testing samples')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--report_wandb', action='store_true', default=True, help='Report to Weights & Biases')
    parser.add_argument('--save_model', action='store_true', default=True, help='Save the trained model')
    parser.add_argument('--save_stats', action='store_true', default=True, help='Save training statistics')
    parser.add_argument('--save_f_maps', action='store_true', help='Save feature maps')
    parser.add_argument('--seed_val', type=int, default=42, help='Random seed')
    parser.add_argument('--tau', type=int, default=4, help='time constant')
    parser.add_argument('--dropout', type = float, default = 0.5, help = 'dropout rate')
    parser.add_argument('--test_var', type = str, default=None, help = 'testing')

    args = parser.parse_args()
    
    if args.sal_type == "None":
        args.sal_type = None

    print(f"Running experiment with dropout rate: {args.dropout}")
    # Run the experiment with the specified parameters
    train_stats = run_experiment(dataset=args.dataset,
                    sal_model=args.sal_model,
                    sal_type=args.sal_type,
                    sal_method=args.sal_method,
                    model_name=args.model_name,
                    wandb_name=args.wandb_name + str(int(args.dropout * 100)),
                    run_outputs_path=args.run_outputs_path,
                    optimizer_name=args.optimizer_name,
                    scheduler_name=args.scheduler_name,
                    num_workers=args.num_workers,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    nTrain=args.nTrain,
                    nTest=args.nTest,
                    learning_rate=args.learning_rate,
                    report_wandb=args.report_wandb,
                    save_model=args.save_model,
                    save_stats=args.save_stats,
                    save_f_maps=args.save_f_maps,
                    seed_val=args.seed_val,
                    k = args.dropout,
                    tau = args.tau)
if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)
    main()
