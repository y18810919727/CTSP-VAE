#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch


from lib import utils
from torch import nn
from lib.ode_func import ODEFunc_w_Poisson, ODEFunc, CDEFunc, CDEFunc_w_Poisson, DEFunc_sp_prior
from lib.encoder_decoder import Encoder_z0_RNN, Encoder_z0_ODE_RNN, Decoder
from lib.diffeq_solver import DiffeqSolver
from lib.stochastic_process_prior import GaussPriorNonlinearAdaptive, GaussPriorNonlinearConstant, GaussPriorLinear
from lib.sp_vae import SPVAE

def create_spvae_model(args, input_dim, z0_prior, obsrv_std, device,
                       classif_per_tp = False, n_labels = 1):

    dim = args.latents
    gen_de_func = None
    if args.posterior == 'ode':
        if args.poisson:
            lambda_net = utils.create_net(dim, input_dim,
                                          n_layers = 1, n_units = args.units, nonlinear = nn.Tanh)

            # ODE function produces the gradient for latent state and for poisson rate
            ode_func_net = utils.create_net(dim * 2, args.latents * 2,
                                            n_layers = args.gen_layers, n_units = args.units, nonlinear = nn.Tanh)

            gen_de_func = ODEFunc_w_Poisson(
                input_dim = input_dim,
                latent_dim = args.latents * 2,
                ode_func_net = ode_func_net,
                lambda_net = lambda_net,
                device = device).to(device)
        else:
            dim = args.latents
            ode_func_net = utils.create_net(dim, args.latents,
                                            n_layers = args.gen_layers, n_units = args.units, nonlinear = nn.Tanh)

            gen_de_func = ODEFunc(
                input_dim = input_dim,
                latent_dim = args.latents,
                ode_func_net = ode_func_net,
                device = device).to(device)
    elif args.posterior == 'cde':
        if args.poisson:
            lambda_net = utils.create_net(dim, input_dim,
                                          n_layers = 1, n_units = args.units, nonlinear = nn.Tanh)

            # # ODE function produces the gradient for latent state and for poisson rate
            # ode_func_net = utils.create_net(dim * 2, args.latents * 2,
            #                                 n_layers = args.gen_layers, n_units = args.units, nonlinear = nn.Tanh)

            gen_de_func = CDEFunc_w_Poisson(
                input_dim = input_dim,
                latent_dim = args.latents * 2,
                lambda_net = lambda_net,
                device = device).to(device)
        else:
            gen_de_func = CDEFunc(
                input_dim = input_dim,
                latent_dim = args.latents,
                device = device).to(device)

    n_rec_dims = args.rec_dims
    enc_input_dim = int(input_dim) * 2 # we concatenate the mask
    gen_data_dim = input_dim

    z0_dim = args.latents
    if args.poisson:
        z0_dim += args.latents # predict the initial poisson rate

    # z0_encoder is used to generate the distribution of z0
    if args.z0_encoder == "odernn":
        ode_func_net = utils.create_net(n_rec_dims, n_rec_dims,
                                        n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)
        rec_ode_func = ODEFunc(
            input_dim = enc_input_dim,
            latent_dim = n_rec_dims,
            ode_func_net = ode_func_net,
            device = device).to(device)

        z0_diffeq_solver = DiffeqSolver(enc_input_dim, rec_ode_func, "euler", args.latents,
                                        odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

        encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver,
                                        z0_dim = z0_dim, n_gru_units = args.gru_units, device = device).to(device)
    elif args.z0_encoder == "rnn":
        encoder_z0 = Encoder_z0_RNN(z0_dim, enc_input_dim,
                                    lstm_output_size = n_rec_dims, device = device).to(device)
    elif args.z0_encoder == "normal":
        raise NotImplementedError
        # class Normal_z0_encoder(nn.Module):
        #     def __init__(self, z0_dim, device):
        #
        # encoder_z0 = lambda data, tps, **kwargs:
    else:
        raise Exception("Unknown encoder for Latent ODE model: " + args.z0_encoder)

    decoder = Decoder(args.latents, gen_data_dim).to(device)

    # "glc (linear model with adjustable covariance), "
    # "gac (affine model with adjustable covariance "
    # "gaa (affine model with adaptive covariance "
    if args.prior == 'glc':
        ct_sp_latent_prior = GaussPriorLinear(latent_dim=z0_dim)
    elif args.prior == 'gac':
        ct_sp_latent_prior = GaussPriorNonlinearConstant(latent_dim=z0_dim)
    elif args.prior == 'gaa':
        ct_sp_latent_prior = GaussPriorNonlinearAdaptive(latent_dim=z0_dim)

    aug_de_func = DEFunc_sp_prior(gen_de_func, ct_sp_latent_prior)
    """
     Aug_func solves the gradient for three parts:
     1. latent
     2. poisson state (optional)
     3. the log probability of dy_dt, including latent and poisson state, in gauss prior at instant time t.
    """

    # The first and forth parameter are unused
    diffeq_solver = DiffeqSolver(gen_data_dim, aug_de_func, 'dopri5', z0_dim + 1,
                                 odeint_rtol=1e-3, odeint_atol=1e-4, device=device)  # TODO adjusting atol and rtol

    prior_ode_solver = DiffeqSolver(gen_data_dim, ct_sp_latent_prior, 'dopri5', z0_dim,
                                    odeint_rtol=1e-4, odeint_atol=1e-5, device=device)
    model = SPVAE(
        input_dim = gen_data_dim,
        latent_dim = args.latents,
        encoder_z0 = encoder_z0,
        decoder = decoder,
        diffeq_solver = diffeq_solver,
        prior_ode_solver=prior_ode_solver,
        z0_prior = z0_prior,
        device = device,
        obsrv_std = obsrv_std,
        use_poisson_proc = args.poisson,
        use_binary_classif = args.classif,
        linear_classifier = args.linear_classif,
        classif_per_tp = classif_per_tp,
        n_labels = n_labels,
        sp_prior=ct_sp_latent_prior,
        train_classif_w_reconstr = (args.dataset == "physionet")
    ).to(device)
    # model = SPVAE(
    #     input_dim = gen_data_dim,
    #     latent_dim = args.latents,
    #     encoder_z0 = encoder_z0,
    #     decoder = decoder,
    #     diffeq_solver = diffeq_solver,
    #     z0_prior = z0_prior,
    #     device = device,
    #     obsrv_std = obsrv_std,
    #     use_poisson_proc = args.poisson,
    #     use_binary_classif = args.classif,
    #     linear_classifier = args.linear_classif,
    #     classif_per_tp = classif_per_tp,
    #     n_labels = n_labels,
    #     train_classif_w_reconstr = (args.dataset == "physionet")
    # ).to(device)

    return model
