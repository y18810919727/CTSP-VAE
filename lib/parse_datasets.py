#!/usr/bin/python
# -*- coding:utf8 -*-

import torch
# from physionet import PhysioNet, variable_time_collate_fn, get_data_min_max
# from person_activity import PersonActivity, variable_time_collate_fn_activity

import lib.utils as utils
import numpy as np
# from generate_timeseries import Periodic_1d
from torch.distributions import uniform

from torch.utils.data import DataLoader
from dataset.mujoco_physics import HopperPhysics
from dataset.generate_timeseries import Periodic_1d
# from physionet import PhysioNet, variable_time_collate_fn, get_data_min_max
# from person_activity import PersonActivity, variable_time_collate_fn_activity

from sklearn import model_selection
import random

#####################################################################################################
def parse_datasets(args, device):

    def basic_collate_fn(batch, time_steps, args=args, data_type='train'):
        batch = torch.stack(batch)
        data_dict = {
            'data':batch,
            'time_steps': time_steps
        }
        data_dict = utils.split_and_subsample_batch(data_dict, args, data_type=data_type)
        return data_dict

    dataset_name = args.dataset
    n_total_tp = args.timepoints + args.extrap
    max_t_extrap = args.max_t / args.timepoints * n_total_tp

    if dataset_name == 'hopper':
        dataset_obj = HopperPhysics(root='data', download=True, generate=False, device=device)
        dataset = dataset_obj.get_dataset()[:args.n]
        dataset = dataset.to(device)

        n_tp_data = dataset[:].shape[1]

        time_steps = torch.arange(start=0, end=n_tp_data, step=1).float().to(device)
        time_steps = time_steps / len(time_steps)
        dataset = dataset.to(device)
        time_steps = time_steps.to(device)

        if not args.extrap:

            n_traj = len(dataset)
            n_tp_data = dataset.shape[1]
            n_reduced_tp = args.timepoints
            start_ind = np.random.randint(0, high=n_tp_data - n_reduced_tp +1, size=n_traj)
            end_ind = start_ind + n_reduced_tp
            sliced = []
            for i in range(n_traj):
                sliced.append(dataset[i, start_ind[i], end_ind[i], :])
            dataset = torch.stack(sliced).to(device)
            time_steps = time_steps[:n_reduced_tp]

        train_y, test_y = utils.split_train_test(dataset, train_fraq=0.8)

        n_samples = len(dataset)
        input_dim = dataset.size(-1)
        batch_size = min(args.batch_size, args.n)
        train_dataloader = DataLoader(train_y, batch_size=batch_size, shuffle=False,
                                      collate_fn=lambda batch: basic_collate_fn(batch, time_steps, data_type="train"))
        test_dataloader = DataLoader(test_y, batch_size=n_samples, shuffle=False,
                                     collate_fn=lambda batch: basic_collate_fn(batch, time_steps, data_type="test"))

        data_objects = {"dataset_obj": dataset_obj,
                        "train_dataloader": utils.inf_generator(train_dataloader),
                        "test_dataloader": utils.inf_generator(test_dataloader),
                        "input_dim": input_dim,
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader)}
        return data_objects

    ########### 1d datasets ###########

    # Sampling args.timepoints time points in the interval [0, args.max_t]
    # Sample points for both training sequence and explapolation (test)
    distribution = uniform.Uniform(torch.Tensor([0.0]),torch.Tensor([max_t_extrap]))
    time_steps_extrap =  distribution.sample(torch.Size([n_total_tp-1]))[:,0]
    time_steps_extrap = torch.cat((torch.Tensor([0.0]), time_steps_extrap))
    time_steps_extrap = torch.sort(time_steps_extrap)[0]

    dataset_obj = None
    ##################################################################
    # Sample a periodic function
    if dataset_name == "periodic":
        dataset_obj = Periodic_1d(
            init_freq = None, init_amplitude = 1.,
            final_amplitude = 1., final_freq = None,
            z0 = 1.)

        dataset = dataset_obj.sample_traj(time_steps_extrap, n_samples = args.n,
                                          noise_weight = args.noise_weight)

        # Process small datasets
        dataset = dataset.to(device)
        time_steps_extrap = time_steps_extrap.to(device)

        train_y, test_y = utils.split_train_test(dataset, train_fraq = 0.8)

        n_samples = len(dataset)
        input_dim = dataset.size(-1)

        batch_size = min(args.batch_size, args.n)
        train_dataloader = DataLoader(train_y, batch_size = batch_size, shuffle=False,
                                      collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "train"))
        test_dataloader = DataLoader(test_y, batch_size = args.n, shuffle=False,
                                     collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "test"))

        data_objects = {#"dataset_obj": dataset_obj,
            "train_dataloader": utils.inf_generator(train_dataloader),
            "test_dataloader": utils.inf_generator(test_dataloader),
            "input_dim": input_dim,
            "n_train_batches": len(train_dataloader),
            "n_test_batches": len(test_dataloader)}

        return data_objects


