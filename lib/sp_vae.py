#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from lib.base_models import VAE_Baseline
from lib import utils
from torch.distributions.normal import Normal
from lib.encoder_decoder import Encoder_z0_ODE_RNN, Encoder_z0_RNN
from lib.interpolate import NaturalCubicSpline, natural_cubic_spline_coeffs
from torch.distributions import kl_divergence, Independent
from lib.likelihood_eval import compute_multiclass_CE_loss, compute_binary_CE_loss, compute_poisson_proc_likelihood

class SPVAE(VAE_Baseline):
    def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver,prior_ode_solver,
        z0_prior, device, obsrv_std = 0.01,
        use_binary_classif = False, use_poisson_proc = False,
        linear_classifier = False,
        classif_per_tp = False,
        n_labels = 1,
        sp_prior=None,
        aug_de_func=None,
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
        self.prior_ode_solver = prior_ode_solver
        self.decoder = decoder
        self.use_poisson_proc = use_poisson_proc
        self.sp_prior = sp_prior

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps,
                           mask = None, n_traj_samples = 1, run_backwards = True, mode='interp'):
        """

        :param time_steps_to_predict:
        :param truth:
        :param truth_time_steps:
        :param mask:
        :param n_traj_samples:
        :param run_backwards:
        :param mode: extrap or interp
        :return:
        """

        if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
                isinstance(self.encoder_z0, Encoder_z0_RNN):

            truth_w_mask = truth
            if mask is not None:
                truth_w_mask = torch.cat((truth, mask), -1)
            first_point_mu, first_point_std = self.encoder_z0(
                truth_w_mask, truth_time_steps, run_backwards = run_backwards)

            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))

        first_point_std = first_point_std.abs()
        assert(torch.sum(first_point_std < 0) == 0.)

        n_traj_samples, n_traj, n_dims = first_point_enc.size()
        if self.use_poisson_proc:
            # append a vector of zeros to compute the integral of lambda
            zeros = torch.zeros([n_traj_samples, n_traj,self.input_dim]).to(utils.get_device(truth))
            first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
            means_z0_aug = torch.cat((means_z0, zeros), -1)
        else:
            first_point_enc_aug = first_point_enc
            means_z0_aug = means_z0

        assert(not torch.isnan(time_steps_to_predict).any())
        assert(not torch.isnan(first_point_enc).any())
        assert(not torch.isnan(first_point_enc_aug).any())

        # Add log prob part of prior in initial state in posterior ODE
        cum_log_prob_prior_0 = torch.zeros([n_traj_samples, n_traj, 1]).to(utils.get_device(truth))
        first_point_enc_aug = torch.cat((first_point_enc_aug, cum_log_prob_prior_0), -1)

        # region make cubic spline interpolation
        truth_for_interpolate = truth.clone()
        if mask is not None:  # NaturalCubicSpline requires tagging nan for unknown positions in data series.
            truth_for_interpolate[mask == 0] = torch.tensor(float('nan')).to(utils.get_device(truth))

        coeffs = natural_cubic_spline_coeffs(truth_time_steps, truth_for_interpolate)
        interpolation = NaturalCubicSpline(truth_time_steps, coeffs)
        self.diffeq_solver.ode_func.splines_setup(interpolation)
        # endregion

        if mode == 'interp':
            sol_y_with_logprob = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)
            sol_y = sol_y_with_logprob[..., :-1]

            # TODO :此处 * const_for_prior_logp是因为：从随机过程对应分布得到的概率密度数值不稳定，因此算ode的时候直接/const_for_prior_logp是因为了，算完之后要乘回来
            prior_log_prob = sol_y_with_logprob[..., -1] * self.diffeq_solver.ode_func.const_for_prior_logp

        elif mode == 'extrap':
            sol_y_with_logprob = self.diffeq_solver(first_point_enc_aug, truth_time_steps)
            sol_y = sol_y_with_logprob[..., :-1]
            # TODO : 利用GaussPrior以sol_y[:,:,-1]为起点，做time_steps_to_predict上的自回归预测:
            # TODO: 方案1: 可以直接从mean采样求解ODE，不过这样就不是采样了。
            # TODO: 方案2：局部线性化(一阶泰勒展开)，在t时刻算t+dt的预测分布并采样。具体公式参考
            # Chen, F., Agüero, J. C., Gilson, M., Garnier, H., & Liu, T. (2017).
            # EM-based identification of continuous-time ARMA Models from irregularly sampled data.
            # Automatica, 77, 293–301. https://doi.org/10.1016/j.automatica.2016.11.020 中的Equ.(6-8)
            raise NotImplementedError

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

            assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
            assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

        pred_x = self.decoder(sol_y)

        all_extra_info = {
            "first_point": (first_point_mu, first_point_std, first_point_enc),
            "latent_traj": sol_y.detach(),
            'prior_log_prob': prior_log_prob
        }

        if self.use_poisson_proc:
            # intergral of lambda from the last step of ODE Solver
            all_extra_info["int_lambda"] = int_lambda[:,:,-1,:]
            all_extra_info["log_lambda_y"] = log_lambda_y

        if self.use_binary_classif:
            if self.classif_per_tp:
                all_extra_info["label_predictions"] = self.classifier(sol_y)
            else:
                all_extra_info["label_predictions"] = self.classifier(first_point_enc).squeeze(-1)

        return pred_x, all_extra_info

    def compute_all_losses(self, batch_dict, n_traj_samples = 1, kl_coef = 1.):

        # self.sample_traj_from_prior(time_steps_to_predict=batch_dict['tp_to_predict'], n_traj_samples=1)
        pred_y, info = self.get_reconstruction(batch_dict["tp_to_predict"],
                                               batch_dict["observed_data"], batch_dict["observed_tp"],
                                               mask = batch_dict["observed_mask"], n_traj_samples = n_traj_samples,
                                               mode = batch_dict["mode"])

        #print("get_reconstruction done -- computing likelihood")
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_std = fp_std.abs()
        fp_distr = Normal(fp_mu, fp_std)

        assert(torch.sum(fp_std < 0) == 0.)

        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

        if torch.isnan(kldiv_z0).any():
            print(fp_mu)
            print(fp_std)
            raise Exception("kldiv_z0 is Nan!")

        # Mean over number of latent dimensions
        # kldiv_z0 shape: [n_traj_samples, n_traj, n_latent_dims] if prior is a mixture of gaussians (KL is estimated)
        # kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard gaussian (KL is computed exactly)
        # shape after: [n_traj_samples]
        kldiv_z0 = torch.mean(kldiv_z0,(1,2))

        # Compute likelihood of all the points
        rec_likelihood = self.get_gaussian_likelihood(
            batch_dict["data_to_predict"], pred_y,
            mask = batch_dict["mask_predicted_data"])

        prior_log_prob_loss = torch.mean(info['prior_log_prob'], (1, 2))

        mse = self.get_mse(
            batch_dict["data_to_predict"], pred_y,
            mask = batch_dict["mask_predicted_data"])

        pois_log_likelihood = torch.Tensor([0.]).to(utils.get_device(batch_dict["data_to_predict"]))
        if self.use_poisson_proc:
            pois_log_likelihood = compute_poisson_proc_likelihood(
                batch_dict["data_to_predict"], pred_y,
                info, mask = batch_dict["mask_predicted_data"])
            # Take mean over n_traj
            pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

        ################################
        # Compute CE loss for binary classification on Physionet
        device = utils.get_device(batch_dict["data_to_predict"])
        ce_loss = torch.Tensor([0.]).to(device)
        if (batch_dict["labels"] is not None) and self.use_binary_classif:

            if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1):
                ce_loss = compute_binary_CE_loss(
                    info["label_predictions"],
                    batch_dict["labels"])
            else:
                ce_loss = compute_multiclass_CE_loss(
                    info["label_predictions"],
                    batch_dict["labels"],
                    mask = batch_dict["mask_predicted_data"])


        # # IWAE loss
        # loss = - torch.logsumexp(rec_likelihood -  kl_coef * kldiv_z0,0)
        # if torch.isnan(loss):
        #     loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0)

        # region IWAE loss with sp prior loss
        loss = - torch.logsumexp(rec_likelihood -  kl_coef * (kldiv_z0 - prior_log_prob_loss),0)
        if torch.isnan(loss):
            loss = - torch.mean(rec_likelihood - kl_coef * (kldiv_z0 - prior_log_prob_loss),0)
        # endregion

        if self.use_poisson_proc:
            loss = loss - 0.1 * pois_log_likelihood

        if self.use_binary_classif:
            if self.train_classif_w_reconstr:
                loss = loss +  ce_loss * 100
            else:
                loss =  ce_loss

        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(rec_likelihood).detach()
        results["mse"] = torch.mean(mse).detach()
        results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["kl_first_p"] =  torch.mean(kldiv_z0).detach()
        results["prior_log_prob"] = torch.mean(prior_log_prob_loss).detach()
        results["std_first_p"] = torch.mean(fp_std).detach()

        if batch_dict["labels"] is not None and self.use_binary_classif:
            results["label_predictions"] = info["label_predictions"].detach()

        return results

    def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples=1):
        """
        # Todo: 目前是方案二，直接用sp prior的mean求解ODE，下一步计划改进为方案三：泰勒一阶展开，然后做小步长的采样。
        :param time_steps_to_predict:
        :param n_traj_samples:
        :return:
        """

        starting_point_enc = self.z0_prior.sample([n_traj_samples, 1, self.latent_dim]).squeeze(-1)

        starting_point_enc_aug = starting_point_enc
        if self.use_poisson_proc:
            n_traj_samples, n_traj, n_dims = starting_point_enc.size()
            # append a vector of zeros to compute the integral of lambda
            zeros = torch.zeros(n_traj_samples, n_traj,self.input_dim).to(self.device)
            starting_point_enc_aug = torch.cat((starting_point_enc, zeros), -1)

        sol_y = self.prior_ode_solver.sample_traj_from_prior(starting_point_enc_aug, time_steps_to_predict,
                                                          n_traj_samples = n_traj_samples)

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

        return self.decoder(sol_y)
