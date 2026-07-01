import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import resize
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import math

from utils.wrapper import *
from utils.filter import *
import warnings

# suppress only the specific Pydantic repr warning from wandb
warnings.filterwarnings(
    "ignore",
    message=r"The 'repr' attribute with value False was provided to the `Field\(\)` function",
    category=UserWarning,
    module="pydantic._internal._generate_schema"
)


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.data_utils import DataModule, load_config
import matplotlib.pyplot as plt

import wandb
import numpy as np
import pickle
from tqdm import tqdm
import copy
import os
from torch.optim.lr_scheduler import StepLR, ExponentialLR

config = load_config('./utils/config.yaml')

class Trainer:
    def __init__(self, nn_model: nn.Module=None, nn_model_name=None, run_outputs_path=None, sal_model=None, report_wandb=None,
            device='cuda', batch_size: int=100, epochs: int=100, num_workers=1, optimizer_name=None, scheduler_name=None,
                 dataset_name=None, save_model=None, save_stats=None, nTrain=None, 
                 nTest=None, learning_rate: float=1e-2, sal_type = 'batch', sal_method = 'maxima', k = 0.1, seed = 42, **kwargs):
        
        self.k = k # sparsity level
        self.seed = seed
        self.model = nn_model.to(device)
        self.sal_type = sal_type
        self.nn_model_name = nn_model_name
        self.run_outputs_path = run_outputs_path # folder to save results
        self.sal_model = sal_model
        self.report_wandb = report_wandb # True if syncing with wandb
        self.device = device
        self.bs = batch_size
        self.sal_method = sal_method
        self.epochs = epochs
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.save_model = save_model
        self.save_stats = save_stats
        self.nTrain = nTrain
        self.nTest = nTest
        self.lr = learning_rate

        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name

        self.sal_transform = SalRCNNTransform(min_size = [800], max_size=1333,image_mean=[0.485, 0.456, 0.406],image_std=[0.229, 0.224, 0.225])
                    
        print('Using optimizer:', self.optimizer_name)

        self.data_module = DataModule(dataset_name=dataset_name, config=config, num_train=nTrain, 
                                num_test=nTest, batch_size=batch_size, num_workers=num_workers, 
                                sal_type = self.sal_type, saliency_model=sal_model)


        if self.sal_type!= None and not self.sal_model:
            raise ValueError("A saliency model name must be provided for saliency maps.")
        self.data_loader_train, self.data_loader_val = self.data_module.get_data_loaders()        
        print('dataloader train: ', len(self.data_loader_train))

        ### --- MODEL SETUP PIPELINE --- ###
        print("Initializing model...")

        # create a deep copy of the original model before modifying it
        # This is important to preserve the original pre-trained weights.
        print("Creating a copy of the original model to preserve pre-trained weights...")
        original_model = copy.deepcopy(self.model)

        # modifying num_classes for PASCAL VOC
        if self.dataset_name == 'voc':
            print("Modifying num classes for PASCAL VOC dataset...")
            num_classes = 21
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            # smaller images need smaller default anchor boxes. Scaled down by a factor of half.
            self.model.rpn.anchor_generator.sizes = ((16,), (32,), (64,), (128,), (256,))

        if self.sal_type in ('batch', 'noise'): # modifying GeneralizedRCNNTransform to 4 channels
            print("Modifying transform for saliency maps...")
            self.model.transform.image_mean = torch.tensor([0.485, 0.456, 0.406, 0.5])
            self.model.transform.image_std  = torch.tensor([0.229, 0.224, 0.225, 0.5])

            #Modifying Convolutional Layer 1
            print("Modifying Conv Layer 1 for saliency maps...")
            new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

            #Initializing weights for the 4th channel
            print("Initializing weights for saliency maps...")
            original_weights = self.model.backbone.body.conv1.weight.data
            with torch.no_grad():
                new_conv.weight[:, :3] = original_weights
                new_conv.weight[:, 3] = torch.rand(64, 7, 7) * 0.01

            self.model.backbone.body.conv1 = new_conv
        

        # Official initialization
        if self.sal_type == 'mask':
            if self.dataset_name == 'coco':
                sizes = ((32,), (64,), (128,), (256,), (512,))
                aspect_ratios = ((0.5, 1.0, 2.0),) * len(sizes)
            elif self.dataset_name == 'voc':
                sizes = ((16,), (32,), (64,), (128,), (256,))
                aspect_ratios = ((0.5, 1.0, 2.0),) * len(sizes)

            print("Initializing Anchor Generator for saliency maps...")
            anchorgen = SalAnchorGenerator(sizes = sizes, aspect_ratios = aspect_ratios, k = self.k, sal_method = self.sal_method, seed = self.seed)

            print("Initializing Objectness and Regression Head for saliency maps...")
            head = SalRPNHead(k = self.k, sal_method = self.sal_method, seed = self.seed)

            print("Initializing Region Proposal Network for saliency maps...")
            network = SalRegionProposalNetwork(
                anchor_generator=anchorgen,
                head=head,
                fg_iou_thresh=0.7,            
                bg_iou_thresh=0.3,                
                batch_size_per_image=256,       
                positive_fraction=0.5,    
                pre_nms_top_n={'training': 2000, 'testing': 2000},
                post_nms_top_n={'training': 2000, 'testing': 2000},
                nms_thresh=0.7
            )
            self.model.rpn = network

        print("Model setup complete.")
        self.model = self.model.to(self.device) # sending to GPU (cuda)
        print("Model device:", next(self.model.parameters()).device)
        ### --- END MODEL SETUP --- ###   

        ### --- OPTIMIZER SETUP --- ###   

        if optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(nn_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        elif optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported optimizers are 'SGD' and 'Adam'.")
        print('Using optimizer:', self.optimizer)

        if scheduler_name == 'ExponentialLR':
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        elif scheduler_name == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif scheduler_name is None:
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}. Supported schedulers are 'ExponentialLR', 'StepLR', or None.")

        ### --- END OPTIMIZER SETUP --- ###

    def train_es(self, es_params):
        """
        Main training loop.
        """
        min_delta = es_params['min_delta']
        patience = es_params['patience']
        best_loss = float('inf')
        counter = 0

        train_trial = np.zeros((18, self.epochs))
        for epoch in range(self.epochs):
            lr = self.optimizer.param_groups[0]['lr']
            train_stats = self.train_one_epoch(epoch)

            print('Mean train loss during epoch %d: %.6f, learning rate :' % (epoch, sum(train_stats) / len(train_stats)), lr)
            stats = self.evaluate()
            dropout_rate = self.k
            mAP = stats['mAP'] # collected data
            mAP50 = stats['mAP50']
            mAP75 = stats['mAP75']
            mAPL = stats['mAPL']
            mAPM = stats['mAPM']
            mAPS = stats['mAPS']
            loss_cls = stats['loss_cls']
            loss_box = stats['loss_box']
            loss_obj = stats['loss_obj']
            loss_rpn_box = stats['loss_rpn_box']
            mAR1 = stats['mAR_1']
            mAR10 = stats['mAR_10']
            mAR100 = stats['mAR_100']
            mARS = stats['mARS']
            mARM = stats['mARM']
            mARL = stats['mARL']

            val_loss = stats['val_loss']
            print('Val loss: %.6f, mAP: %.6f' % (sum(val_loss)/len(val_loss), sum(mAP)/len(mAP)))

            avg_loss = sum(train_stats) / len(train_stats)
            avg_vloss = sum(val_loss) / len(val_loss)
            avg_mAP = sum(mAP) / len(mAP)

            avg_mAPL = sum(mAPL) / len(mAPL)
            avg_mAPM = sum(mAPM) / len(mAPM)
            avg_mAPS = sum(mAPS) / len(mAPS)
            avg_mAP50 = sum(mAP50) / len(mAP50)
            avg_mAP75= sum(mAP75) / len(mAP75)
            avg_loss_cls = sum(loss_cls) / len(loss_cls)
            avg_loss_box = sum(loss_box) / len(loss_box)
            avg_loss_obj = sum(loss_obj) / len(loss_obj)
            avg_loss_rpn_box = sum(loss_rpn_box) / len(loss_rpn_box)
            avg_mAR1 = sum(mAR1) / len(mAR1)
            avg_mAR10 = sum(mAR10) / len(mAR10)
            avg_mAR100 = sum(mAR100) / len(mAR100)
            avg_mARS = sum(mARS) / len(mARS)
            avg_mARM = sum(mARM) / len(mARM)
            avg_mARL = sum(mARL) / len(mARL)



            train_trial[0,epoch-1] = avg_loss
            train_trial[1,epoch-1] = avg_vloss
            train_trial[2,epoch-1] = avg_mAP
            train_trial[3, epoch-1] = avg_mAP50
            train_trial[4, epoch-1] = avg_mAP75
            train_trial[5, epoch-1] = avg_mAPL
            train_trial[6, epoch-1] = avg_mAPM
            train_trial[7, epoch-1] = avg_mAPS
            train_trial[8, epoch-1] = avg_loss_cls
            train_trial[9, epoch-1] = avg_loss_box
            train_trial[10, epoch-1] = avg_loss_obj
            train_trial[11, epoch-1] = avg_loss_rpn_box
            train_trial[12, epoch-1] = avg_mAR1
            train_trial[13, epoch-1] = avg_mAR10
            train_trial[14, epoch-1] = avg_mAR100
            train_trial[15, epoch-1] = avg_mARS
            train_trial[16, epoch-1] = avg_mARM
            train_trial[17, epoch-1] = avg_mARL

            # log metrics to wandb
            if self.report_wandb:
                # wandb.log({"val_acc": val_acc, "val_loss": avg_vloss, "train_loss": avg_loss})
                wandb.log({"mAP":avg_mAP,"train_loss": avg_loss, "val_loss": avg_vloss, "epoch": epoch, "mAP50" : avg_mAP50, "mAP75" : avg_mAP75, "mAPL" : avg_mAPL, "mAPM" : avg_mAPM, "mAPS" : avg_mAPS, 
                "loss_classifier" : avg_loss_cls, "loss_box_reg" : avg_loss_box, "loss_RPN_objectness" : avg_loss_obj, "loss_RPN_box_reg" : avg_loss_rpn_box, "mAR1" : avg_mAR1, "mAR10" : avg_mAR10, "mAR100" : avg_mAR100,
                "mARS" : avg_mARS, "mARM": avg_mARM, "mARL": avg_mARL, "dropout_rate" : dropout_rate})

            # --- early stopping monitor ---
            if avg_vloss < best_loss - min_delta:
                best_loss = avg_vloss
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        if self.save_stats:
            # save the training trial results
            train_trial_array = np.array(train_trial)
            file_path = self.run_outputs_path + str(self.sal_model) + "/train_stats/"
            os.makedirs(file_path, exist_ok=True)

            # save the training trial results as a numpy array
            np.save(file_path+self.sal_method+str(self.k)+ '_run.npy', train_trial_array)

        # save model weights for reuse and furhter analysis
        if self.save_model:
            # create the directory if it doesn't exist
            file_path = self.run_outputs_path + str(self.sal_model) + "/saved_model_weights/"
            os.makedirs(file_path, exist_ok=True)
            # save the model weights
            torch.save(self.model.state_dict(), file_path+self.sal_method+str(self.k)+'_weights.pt')

        return train_trial
    
    def train_step_sal(self, samples, masks, targets):

        # clear previous gradients
        self.optimizer.zero_grad()
        sal_images, _ = self.sal_transform(masks)
        sal_maps = sal_images.tensors
        self.model.rpn.anchor_generator.sal = sal_maps
        self.model.rpn.head.sal = sal_maps
        img_ids = [d["img_id"] for d in targets]
        self.model.rpn.anchor_generator.img_id = img_ids
        self.model.rpn.head.img_id = img_ids
        # The model's forward pass now only takes the main input.
        loss_dict = self.model(samples, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        self.optimizer.step()

        return losses.item()

    def train_step(self,samples,targets):

        self.optimizer.zero_grad()
        loss_dict = self.model(samples, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        self.optimizer.step()

        return losses.item()
    
    def train_one_epoch(self, epoch):
        self.model.train() # set model to training mode
        loss_values = []
        dropout_rate = self.k
        self.model.rpn.head.k = dropout_rate
        self.model.rpn.anchor_generator.k = dropout_rate

        # loop through the training data batches
        
        if self.sal_type != None:
            for images, masks, targets in tqdm(self.data_loader_train):
                if images is None or targets is None:
                    continue
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                masks = [mask.to(self.device) for mask in masks]
                # train the model on the current batch
                loss_values.append(self.train_step_sal(images, masks, targets))

                # for opt in self.optimizers: opt.step()
                if self.scheduler is not None:
                    self.scheduler.step()


        
        else:
            for images, targets in tqdm(self.data_loader_train):
                if images is None or targets is None:
                    continue
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_values.append(self.train_step(images, targets))
            if self.scheduler is not None:
                self.scheduler.step()

        return loss_values # return the loss values for the epoch
    
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval() # set model to evaluation mode
        stats = {}
        stats['val_loss'] = []
        stats['mAP'] = []
        stats['mAP50'] = []
        stats['mAP75'] = []
        stats['mAPL'] = []
        stats['mAPM'] = []
        stats['mAPS'] = []
        stats['loss_cls'] = []
        stats['loss_box'] = []
        stats['loss_obj'] = []
        stats['loss_rpn_box'] = []
        stats['mAR_1'] = []
        stats['mAR_10'] = []
        stats['mAR_100'] = []
        stats['mARS'] =[]
        stats['mARM'] = []
        stats['mARL'] = []

        # loop through the validation data batches

        if self.sal_type != None:
            for images, masks, targets in tqdm(self.data_loader_val):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                masks = [mask.to(self.device) for mask in masks]
                # perform evaluation step
                mAP, mAP50, mAP75, mAPL, mAPM, mAPS, val_loss, loss_cls, loss_box, loss_obj, loss_rpn_box, mAR1, mAR10, mAR100, mARS, mARM, mARL = self.eval_step_sal(images, masks, targets)

                # new_stats is a tuple of (loss, accuracy), so we need stats to be a list of lists (one for each metric)
                stats['mAP'].append(mAP)
                stats['mAP50'].append(mAP50)
                stats['mAP75'].append(mAP75)
                stats['mAPL'].append(mAPL)
                stats['mAPM'].append(mAPM)
                stats['mAPS'].append(mAPS)
                stats['val_loss'].append(val_loss)
                stats['loss_cls'].append(loss_cls)
                stats['loss_box'].append(loss_box)
                stats['loss_obj'].append(loss_obj)
                stats['loss_rpn_box'].append(loss_rpn_box)
                stats['mAR_1'].append(mAR1)
                stats['mAR_10'].append(mAR10)
                stats['mAR_100'].append(mAR100)
                stats['mARS'].append(mARS)
                stats['mARM'].append(mARM)
                stats['mARL'].append(mARL)
        else:
            for images, targets in tqdm(self.data_loader_val):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                # perform evaluation step
                mAP, mAP50, mAP75, mAPL, mAPM, mAPS, val_loss, loss_cls, loss_box, loss_obj, loss_rpn_box, mAR1, mAR10, mAR100, mARS, mARM, mARL = self.eval_step(images, targets)
                
                stats['mAP'].append(mAP)
                stats['mAP50'].append(mAP50)
                stats['mAP75'].append(mAP75)
                stats['mAPL'].append(mAPL)
                stats['mAPM'].append(mAPM)
                stats['mAPS'].append(mAPS)
                stats['val_loss'].append(val_loss)
                stats['loss_cls'].append(loss_cls)
                stats['loss_box'].append(loss_box)
                stats['loss_obj'].append(loss_obj)
                stats['loss_rpn_box'].append(loss_rpn_box)
                stats['mAR_1'].append(mAR1)
                stats['mAR_10'].append(mAR10)
                stats['mAR_100'].append(mAR100)
                stats['mARS'].append(mARS)
                stats['mARM'].append(mARM)
                stats['mARL'].append(mARL)

        return stats


    @torch.no_grad()
    def eval_step_sal(self, samples, masks, targets):
        self.model.eval()
        sal_images, _ = self.sal_transform(masks)
        sal_maps = sal_images.tensors
        self.model.rpn.anchor_generator.sal = sal_maps
        self.model.rpn.head.sal = sal_maps
        img_ids = [d["img_id"] for d in targets]
        self.model.rpn.anchor_generator.img_id = img_ids
        self.model.rpn.head.img_id = img_ids
        with torch.no_grad():
            outputs = self.model(samples)
        # compute metrics
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools")
        metric.update(outputs, targets)

        results = metric.compute()


        self.model.train()   # required to get loss_dict
        with torch.no_grad():
            loss_dict = self.model(samples, targets)
            val_loss = sum(loss for loss in loss_dict.values())
            loss_cls = loss_dict['loss_classifier']
            loss_box = loss_dict['loss_box_reg']
            loss_obj = loss_dict['loss_objectness']
            loss_rpn_box = loss_dict['loss_rpn_box_reg']

        return results['map'].item(), results['map_50'].item(), results['map_75'].item(), results['map_large'].item(), results['map_medium'].item(), results['map_small'].item(), val_loss, loss_cls, loss_box, loss_obj, loss_rpn_box,results['mar_1'].item(), results['mar_10'].item(), results['mar_100'].item(), results['mar_small'].item(), results['mar_medium'].item(), results['mar_large'].item()

    @torch.no_grad()
    def eval_step(self, samples, targets):
        self.model.eval()  # set model to evaluation mode
        # forward pass: compute loss and predictions
        with torch.no_grad():
            outputs = self.model(samples)
        # compute metrics
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools")

        metric.update(outputs, targets)

        results = metric.compute()

        

        self.model.train()   # required to get loss_dict
        with torch.no_grad():
            loss_dict = self.model(samples, targets)
            val_loss = sum(loss for loss in loss_dict.values())
            loss_cls = loss_dict['loss_classifier']
            loss_box = loss_dict['loss_box_reg']
            loss_obj = loss_dict['loss_objectness']
            loss_rpn_box = loss_dict['loss_rpn_box_reg']

        return results['map'].item(), results['map_50'].item(), results['map_75'].item(), results['map_large'].item(),results['map_medium'].item(), results['map_small'].item(), val_loss, loss_cls, loss_box, loss_obj, loss_rpn_box,results['mar_1'].item(), results['mar_10'].item(), results['mar_100'].item(), results['mar_small'].item(), results['mar_medium'].item(), results['mar_large'].item()
