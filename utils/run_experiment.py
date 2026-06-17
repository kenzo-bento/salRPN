import torch
import copy
import random
import numpy as np
from utils.trainer import Trainer
from utils.run import perform_run
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

def run_experiment(dataset,
                   sal_type='batch',
                   sal_model=None,
                   sal_method='maxima',
                   model_name='resnet',
                   run_outputs_path='./run_outputs',
                   optimizer_name='SGD',
                   wandb_name = 'HardSal 10%',
                   scheduler_name=None,
                   epochs=200,
                   num_workers=1,
                   batch_size=100,
                   nTrain=50000,
                   nTest=10000,
                   learning_rate=2e-2,
                   report_wandb=True,
                   save_model=False,
                   save_stats=False, 
                   save_f_maps=False,
                   seed_val=42,
                   k = 0.1,
                   tau = 8,
                   test_var=None):
    
    # for reproducibility
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device = torch.device('cuda')
    print(device)
    print(torch.cuda.is_available())

    if model_name == 'resnet':
        nn_model = fasterrcnn_resnet50_fpn(weights = None) # using randomly initialized weights
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported models are 'resnet'.")
        
    nn_model.to(device)
    
    trainer = Trainer(
        nn_model=nn_model,
        nn_model_name=model_name,
        run_outputs_path=run_outputs_path,
        sal_method=sal_method,
        sal_model=sal_model,
        report_wandb=report_wandb,
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        num_workers=num_workers,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        dataset_name=dataset,
        sal_type = sal_type,
        save_model=save_model,
        save_stats=save_stats,
        save_f_maps=save_f_maps,
        nTrain=nTrain,
        nTest=nTest,
        learning_rate=learning_rate,
        k = k,
        seed = seed_val,
        tau = tau,
        test_var = test_var,
    )

    base_model = copy.deepcopy(trainer.model)

    # properly detach the base model
    for param in base_model.parameters():
        param.requires_grad = False
        param.data = param.data.clone().detach()  # breaking graph connections

    # also set to eval mode to be safe
    base_model.eval()

    # NOTE: MOST OF THESE ARE NEVER USED IN THE CODE
    es_params = {
        'patience': 10,
        'min_delta': 0.001,
        'ls_init': float('inf'),
        'delta_L_init': 0.1,
        'alpha': 0.1,
        'delayCount_init': 0,
    }

    wandb_params = {
        'learning_rate': learning_rate,
        'optimizer': optimizer_name,
        'scheduler': scheduler_name,
        'model': model_name,
        'sal_model': sal_model,
        'dataset': dataset,
        'epochs': epochs,
        'batch_size': batch_size,
        'nTrain': nTrain,
        'nTest': nTest,
        'min_delta': es_params['min_delta'],
        'patience': es_params['patience'],
        'random_seed' : seed_val,
    }

    train_stats = perform_run(
        trainerObj=trainer,
        base_model=base_model,
        es_params=es_params,
        wandb_params=wandb_params,
        report_wandb=report_wandb,
        wandb_name = wandb_name
    )

    return train_stats
