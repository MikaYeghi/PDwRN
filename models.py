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

import pdb

@META_ARCH_REGISTRY.register()
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1,
                 height=384, width=384,
                 known_n_points=None,
                 ultrasmall=False,
                 device=torch.device('cuda')):
        """
        Instantiate a UNet network.
        :param n_channels: Number of input channels (e.g, 3 for RGB)
        :param n_classes: Number of output classes
        :param height: Height of the input images
        :param known_n_points: If you know the number of points,
                               (e.g, one pupil), then set it.
                               Otherwise it will be estimated by a lateral NN.
                               If provided, no lateral network will be build
                               and the resulting UNet will be a FCN.
        :param ultrasmall: If True, the 5 central layers are removed,
                           resulting in a much smaller UNet.
        :param device: Which torch device to use. Default: CUDA (GPU).
        """
        super(UNet, self).__init__()
        n_channels = 3
        self.ultrasmall = ultrasmall
        self.device = device

        # With this network depth, there is a minimum image size
        if height < 256 or width < 256:
            raise ValueError('Minimum input image size is 256x256, got {}x{}'.\
                             format(height, width))

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        if self.ultrasmall:
            self.down3 = down(256, 512, normaliz=False)
            self.up1 = up(768, 128)
            self.up2 = up(256, 64)
            self.up3 = up(128, 64, activ=False)
        else:
            self.down3 = down(256, 512)
            self.down4 = down(512, 512)
            self.down5 = down(512, 512)
            self.down6 = down(512, 512)
            self.down7 = down(512, 512)
            self.down8 = down(512, 512, normaliz=False)
            self.up1 = up(1024, 512)
            self.up2 = up(1024, 512)
            self.up3 = up(1024, 512)
            self.up4 = up(1024, 512)
            self.up5 = up(1024, 256)
            self.up6 = up(512, 128)
            self.up7 = up(256, 64)
            self.up8 = up(128, 64, activ=False)
        self.outc = outconv(64, n_classes)
        self.out_nonlin = nn.Sigmoid()

        self.known_n_points = known_n_points
        if known_n_points is None:
            steps = 3 if self.ultrasmall else 8
            height_mid_features = height//(2**steps)
            width_mid_features = width//(2**steps)
            self.branch_1 = nn.Sequential(nn.Linear(height_mid_features*\
                                                    width_mid_features*\
                                                    512,
                                                    64),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.5))
            self.branch_2 = nn.Sequential(nn.Linear(height*width, 64),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.5))
            self.regressor = nn.Sequential(nn.Linear(64 + 64, 1),
                                           nn.ReLU())

        # This layer is not connected anywhere
        # It is only here for backward compatibility
        self.lin = nn.Linear(1, 1, bias=False)

    def forward(self, x):

        batch_size = x.shape[0]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        if self.ultrasmall:
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
        else:
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            x7 = self.down6(x6)
            x8 = self.down7(x7)
            x9 = self.down8(x8)
            x = self.up1(x9, x8)
            x = self.up2(x, x7)
            x = self.up3(x, x6)
            x = self.up4(x, x5)
            x = self.up5(x, x4)
            x = self.up6(x, x3)
            x = self.up7(x, x2)
            x = self.up8(x, x1)
        x = self.outc(x)
        x = self.out_nonlin(x)

        # Reshape Bx1xHxW -> BxHxW
        # because probability map is real-valued by definition
        x = x.squeeze(1)

        if self.known_n_points is None:
            middle_layer = x4 if self.ultrasmall else x9
            middle_layer_flat = middle_layer.view(batch_size, -1)
            x_flat = x.view(batch_size, -1)

            lateral_flat = self.branch_1(middle_layer_flat)
            x_flat = self.branch_2(x_flat)

            regression_features = torch.cat((x_flat, lateral_flat), dim=1)
            regression = self.regressor(regression_features)

            return x, regression
        else:
            n_pts = torch.tensor([self.known_n_points]*batch_size,
                                 dtype=torch.get_default_dtype())
            n_pts = n_pts.to(self.device)
            return x, n_pts
        # summ = torch.sum(x)
        # count = self.lin(summ)

        # count = torch.abs(count)

        # if self.known_n_points is not None:
            # count = Variable(torch.cuda.FloatTensor([self.known_n_points]))

        # return x, count

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

        return {
            "loss_cls": loss_cls / normalizer,
            "loss_point_reg": loss_point_reg / normalizer,
        }