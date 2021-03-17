#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

from lib.utils import *
from lib import utils
import sys
from lib.plotting import *

from lib.parse_datasets import parse_datasets
from torch.distributions.normal import Normal

file_name = os.path.basename(__file__)[:-3]
from lib.create_latent_ode_model import create_LatentODE_model
from lib.create_spvae_model import create_spvae_model
from config.config import args, device

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random() * 100000)
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')

    start = time.time()
    print("Sampling dataset of {} training examples".format(args.n))

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)

    utils.makedirs("results/")

    ##################################################################
    data_obj = parse_datasets(args, device)
    input_dim = data_obj["input_dim"]

    classif_per_tp = False
    if ("classif_per_tp" in data_obj):
        # do classification per time point rather than on a time series as a whole
        classif_per_tp = data_obj["classif_per_tp"]

    if args.classif and (args.dataset == "hopper" or args.dataset == "periodic"):
        raise Exception("Classification task is not available for MuJoCo and 1d datasets")

    n_labels = 1
    if args.classif:
        if ("n_labels" in data_obj):
            n_labels = data_obj["n_labels"]
        else:
            raise Exception("Please provide number of labels for classification task")

    ################################################
    obsrv_std = 0.01
    if args.dataset == 'hopper':
        obsrv_std = 1e-3

    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))


    if args.latent_ode:
        model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device,
                                       classif_per_tp=classif_per_tp,
                                       n_labels=n_labels)
    elif args.sp_vae:
        model = crea

    else:
        raise Exception('Model not specified')

    if args.viz:
        viz = Visualizations(device)

    ##################################################################

    #Load checkpoint and evaluate the model
    if args.load is not None:
        utils.get_ckpt_model(ckpt_path, model, device)
        exit()

    ##################################################################
    # Training

    log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)

    optimizer = optim.Adamax(model.parameters(), lr=args.lr)

    num_batches = data_obj["n_train_batches"]

    for itr in range(1, num_batches * (args.niters + 1)):
        optimizer.zero_grad()
        utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)

        wait_until_kl_inc = 10
        if itr // num_batches < wait_until_kl_inc:
            kl_coef = 0.
        else:
            kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))

        batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
        train_res = model.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
        train_res["loss"].backward()
        optimizer.step()

        n_iters_to_viz = 1
        if itr % (n_iters_to_viz * num_batches) == 0:
            with torch.no_grad():

                test_res = compute_loss_all_batches(model,
                                                    data_obj["test_dataloader"], args,
                                                    n_batches = data_obj["n_test_batches"],
                                                    experimentID = experimentID,
                                                    device = device,
                                                    n_traj_samples = 3, kl_coef = kl_coef)

                message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                    itr//num_batches,
                    test_res["loss"].detach(), test_res["likelihood"].detach(),
                    test_res["kl_first_p"], test_res["std_first_p"])

                logger.info("Experiment " + str(experimentID))
                logger.info(message)
                logger.info("KL coef: {}".format(kl_coef))
                logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
                logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))

                if "auc" in test_res:
                    logger.info("Classification AUC (TEST): {:.4f}".format(test_res["auc"]))

                if "mse" in test_res:
                    logger.info("Test MSE: {:.4f}".format(test_res["mse"]))

                if "accuracy" in train_res:
                    logger.info("Classification accuracy (TRAIN): {:.4f}".format(train_res["accuracy"]))

                if "accuracy" in test_res:
                    logger.info("Classification accuracy (TEST): {:.4f}".format(test_res["accuracy"]))

                if "pois_likelihood" in test_res:
                    logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))

                if "ce_loss" in test_res:
                    logger.info("CE loss: {}".format(test_res["ce_loss"]))

            torch.save({
                'args': args,
                'state_dict': model.state_dict(),
            }, ckpt_path)


            # Plotting
            if args.viz:
                with torch.no_grad():
                    test_dict = utils.get_next_batch(data_obj["test_dataloader"])

                    print("plotting....")
                    if isinstance(model, LatentODE) and (args.dataset == "periodic"): #and not args.classic_rnn and not args.ode_rnn:
                        plot_id = itr // num_batches // n_iters_to_viz
                        viz.draw_all_plots_one_dim(test_dict, model,
                                                   plot_name = file_name + "_" + str(experimentID) + "_{:03d}".format(plot_id) + ".png",
                                                   experimentID = experimentID, save=True)
                        plt.pause(0.01)
    torch.save({
        'args': args,
        'state_dict': model.state_dict(),
    }, ckpt_path)









