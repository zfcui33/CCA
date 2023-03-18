import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from .Deit import deit_small_distilled_patch16_224
from .Deit import deit_tiny_distilled_patch16_224
from .Deit import deit_base_distilled_patch16_224
from .Deit import vit_small_patch16_224

class CCA(nn.Module):
    """
    Simple Siamese baseline with avgpool
    """
    def __init__(self,  args, base_encoder=None):
        """
        dim: feature dimension (default: 512)
        """
        super(CCA, self).__init__()
        self.dim = args.dim
        self.size_sat = [256, 256]
        self.size_sat_default = [256, 256]
        self.size_grd = [256, 256]

        if args.sat_res != 0:
            self.size_sat = [args.sat_res, args.sat_res]
        if args.fov != 0:
            self.size_grd[1] = int(args.fov / 360. * self.size_grd[1])

        self.ratio = self.size_sat[0]/self.size_sat_default[0]
        base_model = deit_small_distilled_patch16_224

        self.query_net = base_model(img_size=self.size_grd, num_classes=args.dim)
        self.reference_net = base_model(img_size=self.size_sat, num_classes=args.dim)
        self.mlphead = torch.nn.Sequential()
        self.polar = None
    def forward(self, im_q, im_k, delta=None, atten=None, indexes=None):
        return self.mlphead(self.query_net(im_q)), self.mlphead(self.reference_net(x=im_k, indexes=indexes))
