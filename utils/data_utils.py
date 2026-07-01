import os
import io
import json
import zipfile
import torch
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET

# yaml files used for paths
def load_config(path: str = 'config.yaml') -> dict:
    """
    Loads the YAML configuration file.
    """
    with open(path, 'r') as file:
        return yaml.safe_load(file)

# visualizing labels
def get_category_dict(config:Dict):
    categories = {}
    config = config.get('coco', {})
    ann_file = os.path.join(config['root'], config['ann_val'])
    with open(ann_file, "r") as f:
        coco = json.load(f)
    for idx in coco["categories"]:
        num = idx["id"]
        name = idx["name"]
        categories[num] = name

    return categories

# loading the COCO dataset
class COCODataset(Dataset):
    def __init__(self, config : Dict, sal_type: str = 'batch',
                 sal_model_name: str = None, train: bool = True,
                 transform=None, sal_transform=None):
        """
        This module acceses, loades, and preprocesses images, annotations, and saliency maps for the COCO dataset.
        """
        self.sal_model_name = sal_model_name # name of the saliency model

        self.train = train # True if train set, False if validation set

        self.config = config.get('coco', {}) # accesses yaml file
        if self.train:
            self.img_dir = os.path.join(self.config['root'], self.config['train'])
            self.ann_file = os.path.join(self.config['root'], self.config['ann_train'])
        else:
            self.img_dir = os.path.join(self.config['root'], self.config['val'])
            self.ann_file = os.path.join(self.config['root'], self.config['ann_val'])


        # loading annotations
        with open(self.ann_file, "r") as f:
            coco = json.load(f)
        
        self.images = {img["id"]: img for img in coco["images"]}
        self.img_to_anns = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # keys to access images and annotations
        self.ids = list(self.images.keys()) # not used in this implementation but needed for COCOEval (instead, used torchmetrics)

        self.transform = transform # toTensor+normalize
        self.sal_transform = sal_transform
        self.sal_type = sal_type
        self.sal_data = None

        if self.sal_type != None: # checking for mask/batch/add logic
            # checking for model name for saliency
            if not sal_model_name:
                raise ValueError("`load_saliency` is True, but no `sal_model_name` was provided.")

            self.sal_data = []
            base_sal_folder =  os.path.join(self.config['root'], "sal", sal_model_name) # configure the root to where the saliency models are
            sal_files = "train_maps/" if self.train else "val_maps/"

            self.sal_dir= os.path.join(base_sal_folder, sal_files)




    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        # load image
        img_id = self.ids[index]
        img_info = self.images[img_id]
        img_name = img_info["file_name"]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')


        # load annotations
        anns = self.img_to_anns.get(img_id, [])

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x+w, y+h # changing bounding box format
            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])
        valid_mask = [(b[2] > b[0] and b[3] > b[1]) for b in boxes] # checking boxes have positive height and width
        boxes = [b for b, valid in zip(boxes, valid_mask) if valid]
        labels = [l for l, valid in zip(labels, valid_mask) if valid]
        if len(boxes) == 0:
            return None # if there are no boxes in the image

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {
            'boxes' : boxes,
            'labels' : labels,
            'img_id' : torch.tensor(int(img_id), dtype = torch.int64) # included for seed
        }
        
        if self.transform:
            image = self.transform(image)

        # Can't use images with no positives.

        # load saliency map
        if self.sal_type != None:
            name, ext = os.path.splitext(img_name)
            sal_name = os.path.join(self.sal_dir, f"{name}_{self.sal_model_name}{ext}")
            sal_map = Image.open(sal_name).convert("L")
            sal_tensor = transforms.ToTensor()(sal_map)
        else:
            return image, target

        if self.sal_type == 'batch': # extra weights channel experiment
            x = torch.cat([image, sal_tensor], dim=0)
            return x, target
        elif self.sal_type == 'noise': # control for extra weights channel experiment
            sal_tensor = torch.randn(1, image.shape[1], image.shape[2])
            x = torch.cat([image, sal_tensor], dim = 0)
            return x, target
         
        if self.sal_type == 'add': # a static modulation of the image
            sal_tensor = sal_tensor.expand(3, -1, -1)
            image = image + sal_tensor
        elif self.sal_type == 'multiply': # a different static modulation
            sal_tensor = sal_tensor.expand(3, -1, -1)
            image = image * sal_tensor
        elif self.sal_type == 'mask': # saliency map localization (Official)
            return image, sal_tensor, target
        return image, target

# functionally similar to COCODataset
class VOCDataset(Dataset):
    """
    This module acceses, loades, and preprocesses images, annotations, and saliency maps for the PASCAL VOC dataset.
    """
    def __init__(self, config : Dict, sal_type: str = 'batch',
                 sal_model_name: str = None, train: bool = True,
                 transform=None, sal_transform=None):
        self.sal_model_name = sal_model_name
        self.train = train
        self.config = config.get('voc', {})
        if self.train:
            self.img_dir = os.path.join(self.config['root'],self.config['train'])
            self.ann_file = os.path.join(self.config['root'],self.config['ann_train'])
        else:
            self.img_dir = os.path.join(self.config['root'], self.config['val'])
            self.ann_file = os.path.join(self.config['root'], self.config['ann_val'])

        self.ids = [
            f.split(".")[0]
            for f in os.listdir(self.ann_file)
            if f.endswith(".xml")
        ]

        self.transform = transform
        self.sal_transform = sal_transform
        self.sal_type = sal_type
        self.sal_data = None

        if self.sal_type != None:
            if not sal_model_name:
                raise ValueError("`load_saliency` is True, but no `sal_model_name` was provided.")

            self.sal_data = []
            if self.train:
                base_sal_folder =  os.path.join(self.config['root'], "VOC2012_train_val", "sal", sal_model_name)
            else:
                base_sal_folder = os.path.join(self.config['root'], "VOC2012_test", "sal", sal_model_name)
            sal_files = "train_maps/" if self.train else "val_maps/"

            self.sal_dir= os.path.join(base_sal_folder, sal_files)
    
        self.class_to_id = {
        "person": 1,
        "aeroplane": 2,
        "bicycle": 3,
        "bird": 4,
        "boat": 5,
        "bottle": 6,
        "bus": 7,
        "car": 8,
        "cat": 9,
        "chair": 10,
        "cow": 11,
        "diningtable": 12,
        "dog": 13,
        "horse": 14,
        "motorbike": 15,
        "pottedplant": 16,
        "sheep": 17,
        "sofa": 18,
        "train": 19,
        "tvmonitor": 20
        } # defining PASCAL VOC classes

        self.id_to_class = {k: v for v, k in self.class_to_id.items()}
        

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        #Load image
        img_id = self.ids[index]

        ann_path = os.path.join(self.ann_file, img_id + ".xml")
        boxes, labels = self.parse_voc_xml(ann_path)

        valid_mask = [(b[2] > b[0] and b[3] > b[1]) for b in boxes]
        boxes = [b for b, valid in zip(boxes, valid_mask) if valid]
        labels = [l for l, valid in zip(labels, valid_mask) if valid]
        if len(boxes) == 0:
            return None

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {
            'boxes' : boxes,
            'labels' : labels,
            'img_id' : torch.tensor(int(img_id), dtype = torch.int64)
        }

        img_name = os.path.join(self.img_dir, img_id + ".jpg")
        image = Image.open(img_name).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        if self.sal_type != None:
            sal_name = os.path.join(self.sal_dir, f"{img_id}_{self.sal_model_name}.jpg")
            sal_map = Image.open(sal_name).convert("L")
            sal_tensor = transforms.ToTensor()(sal_map)
        else:
            return image, target

        if self.sal_type == 'batch':
            x = torch.cat([image, sal_tensor], dim=0)
            return x, target
        elif self.sal_type == 'noise':
            sal_tensor = torch.randn(1, image.shape[1], image.shape[2])
            x = torch.cat([image, sal_tensor], dim = 0)
            return x, target
        
        if self.sal_type == 'add':
            sal_tensor = sal_tensor.expand(3, -1, -1)
            image = image + sal_tensor
        elif self.sal_type == 'multiply':
            sal_tensor = sal_tensor.expand(3, -1, -1)
            image = image * sal_tensor
        elif self.sal_type == 'mask':
            return image, sal_tensor, target
        return image,target

    def parse_voc_xml(self, xml_path):
        # loading annotations
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls not in self.class_to_id:
                continue  # ignore unknown classes

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_id[cls])

        return boxes, labels

class DataModule:
    """
    A general module to prepare datasets, collate the batches, and forward to the model for processing.
    """
    def __init__(self, dataset_name: str, config: Dict, num_train: int, num_test: int, batch_size: int,
                 num_workers: int, sal_type: str = 'batch', saliency_model: str = None):
        
        self.dataset_name = dataset_name
        self.config = config
        self.num_train = num_train
        self.num_test = num_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sal_type = sal_type
        self.saliency_model = saliency_model
        self.train_transforms, self.test_transforms, self.saliency_transforms = self._create_transforms()
        self.dataset_train, self.dataset_test = self._create_datasets()
        

    def collate_fn(self, batch):
        # filter out None samples
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None, None

        if self.sal_type == 'mask':
            images, masks, targets = zip(*batch)
            return list(images), masks, list(targets)
        else:
            images, targets = zip(*batch)
            return list(images), list(targets)


    def visualize_samples(self, split: str = 'train', num_samples: int = 4):
        """
        Quick Visualization.
        """
        print(f"Visualizing {num_samples} samples from the '{split}' dataset...")
        if split not in ['train', 'test']:
            print("Error: Split must be 'train' or 'test'.")
            return

        # select the dataset to visualize
        dataset = self.dataset_train if split == 'train' else self.dataset_test

        # getting id to name category dict
        categories = get_category_dict(self.config)

        if not dataset:
            print(f"The {split} dataset is not loaded.")
            return

        # create a figure for the plots
        fig, axes = plt.subplots(num_samples, 2, figsize=(6, 2 * num_samples))
        fig.suptitle(f"Sample Images and Saliency Maps ({split.capitalize()} Set)", fontsize=14)

        color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "black", "magenta","green","aqua"]*10 # different colors for different classes
        color_index = 0
        for i in range(num_samples):

            idx = random.randint(0, len(dataset) - 1)

            item = dataset[idx]
            # skipping images with no bounding boxes
            if item is None:
                continue
            img_tensor, target = item
            if self.sal_type == 'batch':
                img_tensor, target = dataset[idx]
                sal_tensor = img_tensor[3, :, :]
                img_tensor = img_tensor[:3,:, :]
            
            # original image
            ax = axes[i,0]
            ax.axis('off')

            # moving tensor to CPU and permuting for proper dimensions
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            
            # un-normalizing for visualization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1) # Clip values to be between 0 and 1
            
            axes[i,0].imshow(img_np)

            # bounding boxes plotting
            bbox = target['boxes']
            class_name = target['labels']
            for bbox, class_name in zip(bbox, class_name):
                x1, y1, x2, y2 = [int(b) for b in bbox]
                x, y, w, h = x1, y1, x2-x1, y2-y1
                color_ = color_list[color_index]
                color_index+=1
                rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color_, facecolor='none')
                if self.dataset_name == 'coco':
                    name = categories[class_name.item()]

                else:
                    id_to_class = {
                        1: "person",
                        2: "aeroplane",
                        3: "bicycle",
                        4: "bird",
                        5: "boat",
                        6: "bottle",
                        7: "bus",
                        8: "car",
                        9: "cat",
                        10: "chair",
                        11: "cow",
                        12: "diningtable",
                        13: "dog",
                        14: "horse",
                        15: "motorbike",
                        16: "pottedplant",
                        17: "sheep",
                        18: "sofa",
                        19: "train",
                        20: "tvmonitor"
                    } # for labelling purposes
                    name = id_to_class[class_name.item()]
                t_box=axes[i,0].text(x, y, name,  color='red', fontsize=10)
                t_box.set_bbox(dict(boxstyle='square, pad=0',facecolor='white', alpha=0.6, edgecolor='blue'))
                axes[i,0].add_patch(rect)
    
            axes[i,0].axis('off')

            # plotting saliency maps
            
            sal_np = sal_tensor.cpu().squeeze() # Remove channel dim for grayscale
            axes[i,1].imshow(sal_np)
            axes[i,1].set_title("Saliency Map")
            axes[i,1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
        plt.savefig("boxes")


    def _create_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Function to create train and test datasets based on config.
        """
        if not self.config:
            raise KeyError(f"Configuration for '{self.dataset_name}' not found in config file.")

        if self.dataset_name == 'coco':
            dataset_class = COCODataset
        elif self.dataset_name == 'voc':
            dataset_class = VOCDataset

        train_args = {
            'config' : self.config,
            'train': True,
            'sal_type': self.sal_type,     # load saliency map if specified
            'sal_model_name': self.saliency_model,  # pass the model name (can be None)
            'transform': self.train_transforms,
            'sal_transform': self.saliency_transforms,
            }
        test_args = {
            'config' : self.config,
            'train': False,
            'sal_type': self.sal_type,
            'sal_model_name': self.saliency_model,
            'transform': self.test_transforms,
            'sal_transform': self.saliency_transforms,
            }

        dataset_train = dataset_class(**train_args)
        dataset_test = dataset_class(**test_args)
        
        # create subsets and return
        return (
            Subset(dataset_train, range(min(len(dataset_train), self.num_train))),
            Subset(dataset_test, range(min(len(dataset_test), self.num_test)))
        )

    def _create_transforms(self) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
        # the mean and std values are standard for the COCO dataset
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # train and test transforms are the same, containing no random elements
        train_aug = test_aug = sal_aug = transforms.Compose([transforms.ToTensor(),normalize])
            
        return train_aug, test_aug, sal_aug

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Returns the training and validation DataLoaders.
        """
        train_loader = DataLoader(
            self.dataset_train, batch_size=self.batch_size,
            sampler=RandomSampler(self.dataset_train), drop_last=True,
            num_workers=self.num_workers, pin_memory=True, collate_fn = self.collate_fn
        )
        test_loader = DataLoader(
            self.dataset_test, batch_size=self.batch_size * 2,
            sampler=RandomSampler(self.dataset_test), drop_last=False,
            num_workers=self.num_workers, pin_memory=True, collate_fn = self.collate_fn
        )
        return train_loader, test_loader
