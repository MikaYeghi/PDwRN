import torch.nn as nn

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.retinanet import RetinaNet

@META_ARCH_REGISTRY.register()
class PDwRN(RetinaNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # The lines below apply to RetinaNetHead
#         self.head.bbox_pred = nn.Conv2d(
# #             conv_dims[-1], num_anchors * 2, kernel_size=3, stride=1, padding=1
#               256, 18, kernel_size=3, stride=1, padding=1
#         )