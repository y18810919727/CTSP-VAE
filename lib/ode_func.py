#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm
from lib.interpolate import NaturalCubicSpline

import lib.utils as utils


class ODEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_func_net, device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()

        self.input_dim = input_dim
        self.device = device

        utils.init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)


class ODEFunc_w_Poisson(ODEFunc):

    def __init__(self, input_dim, latent_dim, ode_func_net,
                 lambda_net, device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc_w_Poisson, self).__init__(input_dim, latent_dim, ode_func_net, device)

        self.latent_ode = ODEFunc(input_dim = input_dim,
                                  latent_dim = latent_dim,
                                  ode_func_net = ode_func_net,
                                  device = device)

        self.latent_dim = latent_dim
        self.lambda_net = lambda_net
        # The computation of poisson likelihood can become numerically unstable.
        #The integral lambda(t) dt can take large values. In fact, it is equal to the expected number of events on the interval [0,T]
        #Exponent of lambda can also take large values
        # So we divide lambda by the constant and then multiply the integral of lambda by the constant
        self.const_for_lambda = torch.Tensor([100.]).to(device)

    def extract_poisson_rate(self, augmented, final_result = True):
        y, log_lambdas, int_lambda = None, None, None

        assert(augmented.size(-1) == self.latent_dim + self.input_dim)
        latent_lam_dim = self.latent_dim // 2

        if len(augmented.size()) == 3:
            int_lambda  = augmented[:,:,-self.input_dim:]
            y_latent_lam = augmented[:,:,:-self.input_dim]

            log_lambdas  = self.lambda_net(y_latent_lam[:,:,-latent_lam_dim:])
            y = y_latent_lam[:,:,:-latent_lam_dim]

        elif len(augmented.size()) == 4:
            int_lambda  = augmented[:,:,:,-self.input_dim:]
            y_latent_lam = augmented[:,:,:,:-self.input_dim]

            log_lambdas  = self.lambda_net(y_latent_lam[:,:,:,-latent_lam_dim:])
            y = y_latent_lam[:,:,:,:-latent_lam_dim]

        # Multiply the intergral over lambda by a constant
        # only when we have finished the integral computation (i.e. this is not a call in get_ode_gradient_nn)
        if final_result:
            int_lambda = int_lambda * self.const_for_lambda

        # Latents for performing reconstruction (y) have the same size as latent poisson rate (log_lambdas)
        assert(y.size(-1) == latent_lam_dim)

        return y, log_lambdas, int_lambda, y_latent_lam

    def get_ode_gradient_nn(self, t_local, augmented):
        y, log_lam, int_lambda, y_latent_lam = self.extract_poisson_rate(augmented, final_result = False)
        dydt_dldt = self.latent_ode(t_local, y_latent_lam)

        log_lam = log_lam - torch.log(self.const_for_lambda)
        return torch.cat((dydt_dldt, torch.exp(log_lam)),-1)


class CDEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(CDEFunc, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.interpolation = None

        # Equ 3 in Neural Controlled Differential Equations for Irregular Time Series
        self.cde_func = utils.create_net(latent_dim, latent_dim * (input_dim + 1), n_units=10, n_layers=1)

        utils.init_network_weights(self.cde_func)
        # self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        if self.interpolation is None:
            raise Exception('Derivative of spline interpolation should be specified before evaluating the CDE func')

        # region Following the 3. method in Neural Controlled Differential Equations
        f_theta = self.cde_func(y).reshape(
            y.shape[:-1] + (self.latent_dim, self.input_dim + 1)
        )
        dx_dt = self.interpolation.derivative(t_local)
        dxt_dt = torch.cat((
            dx_dt, torch.tensor(1.0).to(utils.get_device(y)).repeat(
                dx_dt.shape[:-1]+(1,)
            )),
            dim=-1
        ).unsqueeze(-1)
        if len(f_theta.shape) == 4:
            dxt_dt = dxt_dt.repeat((f_theta.shape[0], 1, 1, 1))
        return (f_theta @ dxt_dt).squeeze(-1)
        # endregion

        # return self.gradient_net(y)

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)

    def splines_setup(self, interpolation):
        """
        An interpolation module is necessary for providing the derivative of observations dX/ds in CDE evaluation.
        :param interpolation: instance of NaturalCubicSpline
        :return:
        """
        assert isinstance(interpolation, NaturalCubicSpline)
        self.interpolation = interpolation


class CDEFunc_w_Poisson(ODEFunc_w_Poisson):

    def __init__(self, input_dim, latent_dim,
                 lambda_net, device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        nn.Module.__init__(self)

        self.latent_cde = CDEFunc(input_dim = input_dim,
                                  latent_dim = latent_dim,
                                  device = device)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lambda_net = lambda_net

        self.device = device
        # The computation of poisson likelihood can become numerically unstable.
        #The integral lambda(t) dt can take large values. In fact, it is equal to the expected number of events on the interval [0,T]
        #Exponent of lambda can also take large values
        # So we divide lambda by the constant and then multiply the integral of lambda by the constant
        self.const_for_lambda = torch.Tensor([100.]).to(device)

    @property
    def latent_ode(self):
        return self.latent_cde

    def splines_setup(self, interpolation):
        self.latent_cde.splines_setup(interpolation)


class DEFunc_sp_prior(nn.Module):
    def __init__(self, de_func, ct_sp_latent_prior):
        """

        :param de_func: cde func or ode func, which determines the derivative of latent state in posterior model
        :param ct_sp_latent_prior:  the continuous-time stochastic prior which evaluates the log prob of the
        """
        super(DEFunc_sp_prior, self).__init__()
        self.de_fun_type = 'cde' if (isinstance(de_func, CDEFunc_w_Poisson) or isinstance(de_func, CDEFunc)) else 'ode'
        self.de_fun = de_func
        self.ct_sp_latent_prior = ct_sp_latent_prior
        self.const_for_prior_logp = torch.Tensor([100.]).to(self.de_fun.device)

    def splines_setup(self, interpolation):
        if self.de_fun_type == 'cde':
            self.de_fun.splines_setup(interpolation)
        else:
            pass

    def get_ode_gradient_nn(self, t_local, y):
        """

        :param t_local:
        :param y: current state in ODE
        :return:
        """
        dy_dt = self.de_fun.get_ode_gradient_nn(t_local, y[...,:-1])  # exclude the log_prob
        dy_dt_log_prob = self.ct_sp_latent_prior.log_prob_eval(t_local, dy_dt, y[...,:-1]) / self.const_for_prior_logp # reducing numerically unstable.
        return torch.cat((dy_dt, dy_dt_log_prob), dim=-1)

    def forward(self, t_local, y, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def extract_poisson_rate(self, augmented, final_result=True):
        return self.de_fun.extract_poisson_rate(augmented, final_result)
