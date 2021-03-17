#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from lib.base_models import VAE_Baseline
from lib.stochastic_process_prior import GaussPrior

class SPVAE(VAE_Baseline):
    def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver,
        z0_prior, device, obsrv_std = None,
        use_binary_classif = False, use_poisson_proc = False,
        linear_classifier = False,
        classif_per_tp = False,
        n_labels = 1,
        sp_prior=None,
        train_classif_w_reconstr = False):

        super(SPVAE, self).__init__(
            input_dim = input_dim, latent_dim = latent_dim,
            z0_prior = z0_prior,
            device = device, obsrv_std = obsrv_std,
            use_binary_classif = use_binary_classif,
            classif_per_tp = classif_per_tp,
            linear_classifier = linear_classifier,
            use_poisson_proc = use_poisson_proc,
            n_labels = n_labels,
            train_classif_w_reconstr = train_classif_w_reconstr)

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.decoder = decoder
        self.use_poisson_proc = use_poisson_proc

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps,
                           mask = None, n_traj_samples = 1, run_backwards = True, mode = None):

        raise NotImplementedError

    def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples = 1):

        raise NotImplementedError

    def compute_all_losses(self, batch_dict, n_traj_samples = 1, kl_coef = 1.):
        raise NotImplementedError

