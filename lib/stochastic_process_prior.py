#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import math
import os
import json

import torch
from torch import nn
from lib import utils


class GaussPrior(nn.Module):

    def __init__(self, latent_dim, type=None):
        """

        :param latent_dim:
        :param type:
            None: no Gauss Prior
            glc: linear model with constant covariance
            gac: affine model with constant covariance
            gaa: affine model with adaptive covariance
        """
        super(GaussPrior, self).__init__()

        if type is None:
            return
        if type == 'glc':
            pass
        elif type == 'gac':
            pass
        elif type == 'gaa':
            pass
        else:
            raise Exception('Unknown type of Gauss prior: {}'.format(type))

        self.mean_net = None
        self.covariance_net = None
        self.type = type

    def log_prob_eval(self, t_local, dy_dt):

        if not self.type:
            return torch.Tensor([0.0]).to(utils.get_device(dy_dt)).reshape(dy_dt.shape[:-1]).unsqueeze(-1)
        else:

            # Todo Implementation for estimating log_prob of dy_dt according to mean_net and covariance_net
            raise NotImplementedError()

