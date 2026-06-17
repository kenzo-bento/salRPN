import torch
from torch import Tensor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
import torch.nn.functional as F
from utils.filter import *
from typing import Optional

class SalAnchorGenerator(AnchorGenerator):
    """
    Subclassing RCNN's AnchorGenerator to apply saliency selection methods.
    """
    def __init__(self, 
                 sizes=((32,), (64,), (128,)), 
                 aspect_ratios=((0.5, 1.0, 2.0),) * 3,
                 k = 0.1,
                 sal_method = 'maxima',
                 seed = 42):
        super().__init__(sizes, aspect_ratios)
        self.k = k
        self.seed = seed
        self.sal = None # (N, 1, 800, 800) later assigned in epochs
        self.sal_method = sal_method
        self.img_id = None # necessary to set seed


    def grid_anchors(self, strides: list[list[Tensor]], sal_list: list[Tensor], length: int) -> list[Tensor]:
        anchors = []
        cell_anchors = self.cell_anchors
        ids = [x for x in self.img_id for _ in range(length)]
        for maps, strides, base_anchors, img_ids in zip(sal_list, strides, cell_anchors, ids):
            maps = maps.squeeze(0).squeeze(0)
            if self.sal_method == 'maxima':
                ys, xs = maxima(maps, k = self.k, seed = img_ids)
            elif self.sal_method == 'hard_sal':
                ys, xs = hard_sal(maps, k= self.k, seed = img_ids)
            elif self.sal_method == 'hard_sal_low':
                ys, xs = hard_sal_low(maps, k = self.k, seed = img_ids)
            elif self.sal_method == 'soft_sal':
                ys, xs = soft_sal(maps, k = self.k, seed = img_ids)
            elif self.sal_method == 'min_and_max':
                ys, xs = min_and_max(maps, k = self.k, seed = img_ids)
            elif self.sal_method == 'random':
                ys, xs = random(maps, k = self.k, seed = img_ids)
            elif self.sal_method == 'soft_sal_random':
                ys, xs = soft_sal_random(maps, k = self.k, seed = img_ids)
            elif self.sal_method == 'gaussian':
                ys, xs = gaussian_mask(maps, k = self.k, seed = img_ids)
            else:
                return ValueError("Sal_method must be 'hard_sal', 'soft_sal', 'random', or 'gaussian'.")
            ys, xs = ys * strides[0], xs *strides[1]
            # combine them into one tensor of central points
            central_points = torch.stack([xs, ys], dim = 1)
            shifts = torch.cat([central_points, central_points], dim = 1)
            
            # for every (base anchor, output anchor) pair, offset each zero-centered base anchor by the center of the output anchor.
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))  
        return anchors

    
    def forward(self, image_list: ImageList, feature_maps: list[Tensor]) -> list[Tensor]:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        resized_sal_list = []
        
        # get a outer loop of sal_maps and an inner loop of sizes
        for sal_map in self.sal:  # outer loop
            inner_list = []
            sal_map = sal_map.unsqueeze(1)
            for size in grid_sizes:  # inner loop
                resized = F.interpolate(
                    sal_map, size=size, mode='bilinear', align_corners=False
                ).clone()
                inner_list.append(resized)
            resized_sal_list.append(inner_list)
        
        
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [
            [
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // g[0]),
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // g[1]),
            ]
            for g in grid_sizes
        ]
        self.set_cell_anchors(dtype, device)

        anchors: list[list[torch.Tensor]] = []

        # produces a list of lists of Tensors with anchor points list[list[Tensor(#, 4)]]
        for sal_feature_maps in resized_sal_list:
            anchors_in_image = self.grid_anchors(strides, sal_feature_maps, len(grid_sizes))
            anchors.append(anchors_in_image)

        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors

class SalRPNHead(RPNHead):
    """
    Subclassing RCNN's RPNHead module to accomodate saliency selection.
    """
    def __init__(self, in_channels = 256, num_anchors = 3, k = 0.1, sal_method = 'maxima', seed = 42):
        super().__init__(in_channels, num_anchors)
        self.k = k
        self.sal = None
        self.sal_method = sal_method
        self.seed = seed
        self.img_id = None

    def forward(self, x: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        # x is [P2, P3, P4, P5, P6] where P --> (N, C, H, W) and C is 256.
        logits = []
        bbox_reg = []

        grid_sizes = [feature_map.shape[-2:] for feature_map in x]
        anchor_mask = []
        delta_mask = []
        for H, W in grid_sizes:
            # resize all saliency maps to current feature map size
            resized_anchor = []
            resized_delta = []
            for ids, sal in enumerate(self.sal):
                sal = sal.unsqueeze(0)
                r = F.interpolate(sal, size=(H, W), mode='bilinear', align_corners=False)
                r = r.squeeze(0).squeeze(0)
                r = create_saliency_mask(r, k = self.k, sal_method = self.sal_method, seed = self.img_id[ids])
                r = r.unsqueeze(0).unsqueeze(0)
                # expand channels to match feature map channels
                anchor = r.expand(-1, 3, -1, -1)
                delta = r.expand(-1, 12, -1, -1)  # shape (1, C, H, W), hardcoded channels
                resized_anchor.append(anchor)
                resized_delta.append(delta)
            # stack along batch dimension
            anchor_mask.append(torch.cat(resized_anchor, dim=0))
            delta_mask.append(torch.cat(resized_delta, dim=0))


        for feature, anchor_mask, delta_mask in zip(x, anchor_mask, delta_mask):
            t = self.conv(feature)
            l = (self.cls_logits(t) + 1e-8) * anchor_mask
            b = (self.bbox_pred(t) + 1e-8) * delta_mask

            # scaling to keep the sum the same
            scale = 1 / (1-self.k)
            l = l * scale
            b = b * float(1)
            logits.append(l)
            bbox_reg.append(b)

        return logits, bbox_reg
    
class SalRegionProposalNetwork(RegionProposalNetwork):
    def forward(
        self,
        images: ImageList,
        features: dict[str, Tensor],
        targets: Optional[list[dict[str, Tensor]]] = None,
    ) -> tuple[list[Tensor], dict[str, Tensor]]:
        """
        Subclassing RCNN's RegionProposalNetwork to accomodate saliency selection
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        

        num_anchors_per_level = [int(torch.count_nonzero(obj).item() / num_images) for obj in objectness
]

        objectness, pred_bbox_deltas = concat_masked_box_prediction_layers(objectness, pred_bbox_deltas)
        

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses

class SalRCNNTransform(GeneralizedRCNNTransform):
    """
    Subclassing GeneralizedRCNNTransform for transforming saliency maps identically.
    """
    def normalize(self, image):
        return image
