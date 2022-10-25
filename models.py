import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.retinanet import RetinaNet
from detectron2.modeling.box_regression import Box2BoxTransformLinear

from unet_parts import *

###
# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import List, Tuple
import torch
from fvcore.nn import sigmoid_focal_loss_jit
from torch import Tensor, nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import CycleBatchNormList, ShapeSpec, batched_nms, cat, get_norm
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.dense_detector import DenseDetector, permute_to_N_HWA_K  # noqa
###

from losses import _dense_box_regression_loss
from utils import pairwise_inverse_distances

import pdb


@META_ARCH_REGISTRY.register()
class PDwRN(RetinaNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor storing the loss.
                Used during training only. The dict keys are: "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        normalizer = self._ema_update("loss_normalizer", max(num_pos_anchors, 1), 100)

        # classification and regression loss
        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[
            :, :-1
        ]  # no loss for the last (background) class
        loss_cls = sigmoid_focal_loss_jit(
            cat(pred_logits, dim=1)[valid_mask],
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        loss_point_reg = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        # Normalize the loss terms
        class_loss = loss_cls / normalizer
        if self.box_reg_loss_type == "l2_point":
            point_regression_loss = 100 * loss_point_reg
        elif self.box_reg_loss_type == "smooth_l1_point":
            point_regression_loss = loss_point_reg / normalizer
        elif self.box_reg_loss_type == "t_distr_point":
            point_regression_loss = loss_point_reg
        else:
            raise NotImplementedError
        
        return {
            "loss_cls": class_loss,
            "loss_point_reg": point_regression_loss
        }
    
    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]: List of #img tensors. i-th element is a vector of labels whose length is
            the total number of anchors across all feature maps (sum(Hi * Wi * A)).
            Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.

            list[Tensor]: i-th element is a Rx4 tensor, where R is the total number of anchors
            across feature maps. The values are the matched gt boxes for each anchor.
            Values are undefined for those anchors not labeled as foreground.
        """
        # pdb.set_trace()
        anchors = Boxes.cat(anchors)  # Rx4

        gt_labels = []
        matched_gt_boxes = []
        for gt_per_image in gt_instances:
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            # match_quality_matrix = pairwise_inverse_distances(gt_per_image.gt_boxes, anchors)
            matched_idxs, anchor_labels = self.anchor_matcher(match_quality_matrix)
            del match_quality_matrix

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]

                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_labels_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_labels_i[anchor_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes