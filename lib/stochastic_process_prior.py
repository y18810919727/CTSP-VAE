#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import math
import os
import json

import torch
from torch import nn
from lib import utils
from lib.utils import create_net
from lib.ode_func import ODEFunc
from torch.distributions.multivariate_normal import MultivariateNormal


class DiagMultivariateNormal(torch.distributions.multivariate_normal.MultivariateNormal):
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) + (scale_tril is not None) + (precision_matrix is not None) != 1:
            raise ValueError("Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified.")

        loc_ = loc.unsqueeze(-1)  # temporarily add dim on right
        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.scale_tril, loc_ = torch.broadcast_tensors(scale_tril, loc_)
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.covariance_matrix, loc_ = torch.broadcast_tensors(covariance_matrix, loc_)
        else:
            if precision_matrix.dim() < 2:
                raise ValueError("precision_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.precision_matrix, loc_ = torch.broadcast_tensors(precision_matrix, loc_)
        self.loc = loc_[..., 0]  # drop rightmost dim

        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super(MultivariateNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            #self._unbroadcasted_scale_tril = torch.cholesky(covariance_matrix)
            self._unbroadcasted_scale_tril = torch.sqrt(covariance_matrix)
        else:  # precision_matrix is not None
            raise NotImplementedError('Only covariance_matrix or scale_tril may be specified')


class GaussPriorFunc(nn.Module):

    def __init__(self):
        super(GaussPriorFunc, self).__init__()

    def get_ode_gradient_nn(self, t_local, y):
        return self.mean(t_local, y)

    def forward(self, t_local, y, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.mean(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)

    def log_prob_eval(self, t_local, dy_dt, y):
        raise NotImplementedError

    def mean(self, t, y):
        raise NotImplementedError


class GaussPriorLinear(GaussPriorFunc):

    def __init__(self, latent_dim, layers=1):
        """
        linear model with constant covariance
        :param latent_dim:
        """
        super(GaussPriorLinear, self).__init__()

        self.latent_dim = latent_dim
        self.linear = torch.nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.log_var = torch.nn.Parameter(torch.randn(latent_dim))

    def log_prob_eval(self, t_local, dy_dt, y):
        return DiagMultivariateNormal(
            self.mean(t_local, y), torch.diag_embed(torch.exp(self.log_var))
        ).log_prob(dy_dt).unsqueeze(-1)

    def mean(self, t_local, y):
        return y @ self.linear


class GaussPriorNonlinearConstant(GaussPriorFunc):

    def __init__(self, latent_dim, layers=1):
        """
        affine model with constant covariance
        :param latent_dim:
        """
        super(GaussPriorNonlinearConstant, self).__init__()

        self.adaptive_mean = create_net(latent_dim, latent_dim)
        # self.log_prob_f = lambda dy_dt, y: self.adaptive_mean(y) + torch.exp(self.log_var)

        self.latent_dim = latent_dim
        self.adaptive_mean = create_net(latent_dim, latent_dim, layers)
        self.log_var = torch.nn.Parameter(torch.randn(latent_dim))

    def log_prob_eval(self, t_local, dy_dt, y):
        return DiagMultivariateNormal(
            self.mean(t_local, y), torch.diag_embed(torch.exp(self.log_var))
        ).log_prob(dy_dt).unsqueeze(-1)

    def mean(self, t_local, y):
        return self.adaptive_mean(y)


class GaussPriorNonlinearAdaptive(GaussPriorFunc):

    def __init__(self, latent_dim, layers=1):
        """
        affine model with adaptive covariance
        :param latent_dim:
        """
        super(GaussPriorNonlinearAdaptive, self).__init__()

        self.latent_dim = latent_dim
        self.adaptive_mean = create_net(latent_dim, latent_dim, n_layers=layers)
        self.log_var_net = create_net(latent_dim, latent_dim, n_layers=layers)

    def log_prob_eval(self, t_local, dy_dt, y):

        return DiagMultivariateNormal(
            self.mean(t_local, y), torch.diag_embed(torch.exp(self.log_var_net(y)))
        ).log_prob(dy_dt).unsqueeze(-1)

    def mean(self, t_local, y):
        return self.adaptive_mean(y)
