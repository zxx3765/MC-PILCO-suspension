# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL contact:	Diego Romeres (romeres@merl.com)
"""

"""
Superclass for model learning objects and basic implementations
### Main variables ###
- data: (state_samples, input samples)
- gp_list: a list with the gp models used to predict the model evolution
### Main methods ###
- add_data: Add new data
- reinforce_model: train hyperparameters and models
- get_next_state: given (state_t,input_t) returns the next state mean and variance
- get_gp_estimate_from_data: returns gp input output and gp output estimate
### Application-depenedent methods ###
- data_to_gp_IO: maps Data in input-output of the gp
- data_to_gp_input: returns gp input from Data
- data_to_gp_output: returns gp ouput from Data
- get_next_state_from_gp_output: maps gp input output in the next state
- get_gp: initialize the gp
"""

import sys

import torch
import torch.utils.data

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions.normal import Normal

import gpr_lib.GP_prior.GP_prior as GP
import gpr_lib.GP_prior.Sparse_GP as Sparse_GP
import gpr_lib.GP_prior.Stationary_GP as SGP


class Model_learning(torch.nn.Module):
    """
    Model learning Class
    """

    def __init__(
        self,
        num_gp,
        init_dict_list,
        approximation_mode=None,
        approximation_dict=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        flg_norm=False,
    ):
        super(Model_learning, self).__init__()
        # Set model info
        self.num_samples = 0
        self.dtype = dtype
        self.device = device
        self.use_exogenous_inputs = False
        self.init_dict_list = init_dict_list
        # Get the gp list
        self.num_gp = num_gp
        self.alpha_list = [None] * num_gp
        self.m_X_list = [None] * num_gp
        self.K_X_inv_list = [None] * num_gp
        self.gp_inputs_tr_list = [None] * num_gp
        # check approximation
        self.approximation_mode = approximation_mode
        if approximation_mode is None:
            print("EXACT GP INFERENCE SELECTED")
            self.get_gp_estimate = self.get_exact_gp_estimate
        else:
            self.approximation_dict = approximation_dict
            print("GP APPROXIMATION SELECTED")
            print("APPROXIMATION MODE: ", approximation_mode)
            print("APPROXIMATION OPTIONS: ", approximation_dict)
            if approximation_mode == "SOR":
                self.get_gp_estimate = self.get_SOR_gp_estimate
                self.Sigma_SOR_list = [None] * num_gp
                self.reg_indices_SOR_list = [None] * num_gp
            elif approximation_mode == "SOD":
                self.SOD_indices = [None] * num_gp
                self.get_gp_estimate = self.get_SOD_gp_estimate
                self.SOD_threshold_mode = approximation_dict["SOD_threshold_mode"]
                self.SOD_threshold = approximation_dict["SOD_threshold"]
                self.flg_SOD_permutation = approximation_dict["flg_SOD_permutation"]
        # init the GP models
        self.init_gp_models()
        # set normalization parameters
        self.flg_norm = flg_norm
        self.norm_list = [1.0] * self.num_gp

    def init_gp_models(self):
        """
        Init gp models
        """
        self.gp_list = torch.nn.ModuleList(
            [
                self.get_gp(gp_index=gp_index, init_dict=self.init_dict_list[gp_index])
                for gp_index in range(0, self.num_gp)
            ]
        )
        if self.approximation_mode == "SOR":
            self.gp_list = torch.nn.ModuleList([Sparse_GP.get_SOR_GP(gp) for gp in self.gp_list])

    def set_eval_mode(self):
        """
        Set all the gp in eval mode
        """
        for gp in self.gp_list:
            gp.set_eval_mode()

    def set_training_mode(self):
        """
        Set all the gp in traing mode
        """
        for gp in self.gp_list:
            gp.set_training_mode()

    def _data_to_gp_io_with_optional_exogenous(self, states, inputs, exogenous_inputs=None):
        if exogenous_inputs is None:
            return self.data_to_gp_IO(states, inputs)
        return self.data_to_gp_IO(states, inputs, exogenous_inputs=exogenous_inputs)

    def _data_to_gp_input_with_optional_exogenous(self, states, inputs, exogenous_inputs=None):
        if exogenous_inputs is None:
            return self.data_to_gp_input(states=states, inputs=inputs)
        return self.data_to_gp_input(states=states, inputs=inputs, exogenous_inputs=exogenous_inputs)

    def add_data(self, new_state_samples, new_input_samples, new_exogenous_samples=None):
        """
        Transform state_sample and input samples in input-output of the gp and store them
        """
        state_tensor = torch.tensor(new_state_samples, dtype=self.dtype, device=self.device)
        input_tensor = torch.tensor(new_input_samples, dtype=self.dtype, device=self.device)
        if new_exogenous_samples is None:
            exogenous_tensor = None
        else:
            exogenous_tensor = torch.tensor(new_exogenous_samples, dtype=self.dtype, device=self.device)

        if self.num_samples == 0:
            self.dim_state = new_state_samples.shape[1]
            self.gp_inputs, self.gp_output_list = self._data_to_gp_io_with_optional_exogenous(
                state_tensor,
                input_tensor,
                exogenous_tensor,
            )
            self.num_samples = new_state_samples.shape[0]
            _, self.dim_input = new_input_samples.shape
        else:

            new_gp_inputs, new_gp_output_list = self._data_to_gp_io_with_optional_exogenous(
                state_tensor,
                input_tensor,
                exogenous_tensor,
            )
            # update samples set
            self.gp_inputs = torch.cat([self.gp_inputs, (new_gp_inputs)])
            self.gp_output_list = [
                torch.cat([self.gp_output_list[gp_index], new_gp_output_list[gp_index]], 0)
                for gp_index in range(0, self.num_gp)
            ]
            self.num_samples, _ = self.gp_inputs.shape

    def reinforce_model(self, optimization_opt_list=None):
        """
        Optimize GP model
        optimization_opt_list is a list collecting dictionaries with the optimization options
        """
        # initialize the GP models
        self.init_gp_models()
        # train each gp
        for gp_index in range(0, self.num_gp):
            self.train_gp(gp_index=gp_index, optimization_opt_dict=optimization_opt_list[gp_index])
            # pretrain each gp (compute alpha, m_X and K_X_inv)
            with torch.no_grad():
                self.pretrain_gp(gp_index=gp_index)

    def pretrain_gp(self, gp_index):
        """
        Compute traininng estimates and returns alpha and K_X_inv
        """
        # make computations
        if self.approximation_mode is None:
            Y_hat, var, alpha, m_X, K_X_inv = self.gp_list[gp_index].get_estimate(
                X=self.gp_inputs, Y=self.gp_output_list[gp_index], X_test=self.gp_inputs, flg_return_K_X_inv=True
            )
            self.K_X_inv_list[gp_index] = K_X_inv
            self.alpha_list[gp_index] = alpha
            self.m_X_list[gp_index] = m_X
            self.gp_inputs_tr_list[gp_index] = self.gp_inputs
        elif self.approximation_mode == "SOD":
            # get the threshold
            if self.SOD_threshold_mode == "relative":
                threshold = self.SOD_threshold * torch.sqrt(self.gp_list[gp_index].get_sigma_n_2())
            elif self.SOD_threshold_mode == "absolute":
                threshold = self.SOD_threshold[gp_index]
            # get the SOD
            self.SOD_indices[gp_index] = self.gp_list[gp_index].get_SOD(
                X=self.gp_inputs,
                Y=self.gp_output_list[gp_index],
                threshold=threshold,
                flg_permutation=self.flg_SOD_permutation,
            )
            # compute the posterior
            Y_hat, var, alpha, m_X, K_X_inv = self.gp_list[gp_index].get_estimate(
                X=self.gp_inputs[self.SOD_indices[gp_index], :],
                Y=self.gp_output_list[gp_index][self.SOD_indices[gp_index], :],
                X_test=self.gp_inputs,
                flg_return_K_X_inv=True,
            )
            self.K_X_inv_list[gp_index] = K_X_inv
            self.alpha_list[gp_index] = alpha
            self.m_X_list[gp_index] = m_X
            self.gp_inputs_tr_list[gp_index] = self.gp_inputs[self.SOD_indices[gp_index], :]
        elif self.approximation_mode == "SOR":
            Y_hat, var, alpha, m_X, Sigma = self.gp_list[gp_index].get_SOR_estimate(
                X=self.gp_inputs, Y=self.gp_output_list[gp_index], X_test=self.gp_inputs, flg_return_Sigma=True
            )
            self.Sigma_SOR_list[gp_index] = Sigma
            self.alpha_list[gp_index] = alpha
            self.m_X_list[gp_index] = m_X
            self.gp_inputs_tr_list[gp_index] = self.gp_inputs
        print("MSE gp " + str(gp_index) + ": ", torch.mean((self.gp_output_list[gp_index] - Y_hat) ** 2))

    def get_next_state(self, current_state, current_input, particle_pred=True, exogenous_input=None):
        """
        Predict the next state given the the current state-input (batches supported).
        Method returns next state samples, together with mean and variance of the gp prediction
        """
        # Get the gp estimate
        _, _, gp_output_mean_list, gp_output_var_list = self.get_one_step_gp_out(
            states=current_state, inputs=current_input, exogenous_input=exogenous_input
        )

        for i in range(self.num_gp):
            gp_output_var_list[i] = gp_output_var_list[i] * self.norm_list[i] ** 2
        # Get the next state form gp IO and return
        return self.get_next_state_from_gp_output(
            current_state=current_state,
            current_input=current_input,
            gp_output_mean_list=gp_output_mean_list,
            gp_output_var_list=gp_output_var_list,
            particle_pred=particle_pred,
        )

    def get_one_step_gp_out(self, states, inputs, exogenous_input=None):
        """
        Compute input-output of the gp and performs estimation:
        The function returns (gp_inputs, gp_outputs, gp_mean_hat, gp_var_hat)
        """
        gp_inputs = self._data_to_gp_input_with_optional_exogenous(
            states=states, inputs=inputs, exogenous_inputs=exogenous_input
        )
        gp_outputs_list = None
        # get gp estimates
        gp_output_mean_list, gp_output_var_list = self.get_gp_estimate(
            gp_inputs=gp_inputs, gp_index_list=range(0, self.num_gp)
        )
        return gp_inputs, gp_outputs_list, gp_output_mean_list, gp_output_var_list

    def get_gp_estimate_from_data(
        self, states, inputs, flg_pretrain=False, gp_index_list=None, flg_onestep=False, exogenous_inputs=None
    ):
        """
        Compute input-output of the gp and performs estimation:
        The function returns (gp_inputs, gp_outputs, gp_mean_hat, gp_var_hat)
        """
        # check index list
        if gp_index_list is None:
            gp_index_list = range(0, self.num_gp)
        if flg_onestep:  # get one-step gp estimates
            gp_inputs = self._data_to_gp_input_with_optional_exogenous(
                states=states, inputs=inputs, exogenous_inputs=exogenous_inputs
            )
            gp_outputs_list = None
        else:  # get the input-output of the gp
            gp_inputs, gp_outputs_list = self._data_to_gp_io_with_optional_exogenous(
                states=states, inputs=inputs, exogenous_inputs=exogenous_inputs
            )
        # pretrain gp
        if flg_pretrain:
            for gp_index in gp_index_list:
                self.pretrain_gp(gp_index=gp_index)
        # get gp estimates
        gp_output_mean_list, gp_output_var_list = self.get_gp_estimate(gp_inputs=gp_inputs, gp_index_list=gp_index_list)
        return gp_inputs, gp_outputs_list, gp_output_mean_list, gp_output_var_list

    def get_exact_gp_estimate(self, gp_inputs, gp_index_list=None):
        """
        Return the gp ouput (mean and variance)
        """
        # check gp_index_list
        if gp_index_list is None:
            gp_index_list = range(0, self.num_gp)
        # initilize the output lists
        gp_output_mean_list = [None] * self.num_gp
        gp_output_var_list = [None] * self.num_gp

        # return gp_output_mean_list, gp_output_var_list
        est_list = [
            self.gp_list[i].get_estimate_from_alpha(
                X=self.gp_inputs_tr_list[i],
                X_test=gp_inputs,
                alpha=self.alpha_list[i],
                m_X=self.m_X_list[i],
                K_X_inv=self.K_X_inv_list[i],
            )
            for i in gp_index_list
        ]
        gp_output_mean_list = [e[0] for e in est_list]
        gp_output_var_list = [e[1].reshape([-1, 1]) for e in est_list]
        return gp_output_mean_list, gp_output_var_list

    def get_SOR_gp_estimate(self, gp_inputs, gp_index_list=None):
        """
        Return the gp ouput (mean and variance)
        """
        # check gp_index_list
        if gp_index_list is None:
            gp_index_list = range(0, self.num_gp)
        # initilize the output lists
        gp_output_mean_list = [None] * self.num_gp
        gp_output_var_list = [None] * self.num_gp
        # get mean and variance estimation
        for gp_index in gp_index_list:
            gp_output_mean_list[gp_index], gp_output_var_list[gp_index] = self.gp_list[
                gp_index
            ].get_SOR_estimate_from_alpha(
                X_test=gp_inputs,
                SOR_alpha=self.alpha_list[gp_index],
                m_X=self.m_X_list[gp_index],
                Sigma=self.Sigma_SOR_list[gp_index],
            )
            # same shape for mean and var
            gp_output_var_list[gp_index] = gp_output_var_list[gp_index].reshape([-1, 1])
        return gp_output_mean_list, gp_output_var_list

    def get_SOD_gp_estimate(self, gp_inputs, gp_index_list):
        """
        Return the gp ouput (mean and variance)
        """
        # initilize the output lists
        gp_output_mean_list = [None] * len(gp_index_list)
        gp_output_var_list = [None] * len(gp_index_list)

        # get mean and variance estimation
        est_list = [
            self.gp_list[i].get_estimate_from_alpha(
                X=self.gp_inputs_tr_list[i],
                X_test=gp_inputs,
                alpha=self.alpha_list[i],
                m_X=self.m_X_list[i],
                K_X_inv=self.K_X_inv_list[i],
            )
            for i in gp_index_list
        ]
        gp_output_mean_list = [e[0] for e in est_list]
        gp_output_var_list = [e[1].reshape([-1, 1]) for e in est_list]
        return gp_output_mean_list, gp_output_var_list

    def get_L1_gp_estimate(self, gp_inputs, gp_index_list=None):
        """
        Return the gp ouput (mean and variance) computed with
        the alpha with minimal L1 norm
        """
        # check gp_index_list
        if gp_index_list is None:
            gp_index_list = range(0, self.num_gp)
        # initilize the output lists
        gp_output_mean_list = [None] * self.num_gp
        gp_output_var_list = [None] * self.num_gp
        # get mean and variance estimation
        for gp_index in gp_index_list:
            gp_output_mean_list[gp_index], gp_output_var_list[gp_index] = self.gp_list[
                gp_index
            ].get_estimate_from_alpha(
                X=self.gp_inputs_tr_list[gp_index][self.alpha_indices_L1_list[gp_index], :],
                X_test=gp_inputs,
                alpha=self.alpha_list[gp_index],
                m_X=self.m_X_list[gp_index],
                K_X_inv=self.K_X_inv_list[gp_index],
            )
            # same shape for mean and var
            gp_output_var_list[gp_index] = gp_output_var_list[gp_index].reshape([-1, 1])
        return gp_output_mean_list, gp_output_var_list

    def to(self, device):
        """
        Move the model parameters to 'device'
        """
        super(Model_learning, self).to(device)
        self.device = device
        for gp in self.gp_list:
            gp.to(device)

    def print_model(self):
        """
        Print the model
        """
        for gp_index, gp in enumerate(self.gp_list):
            print("GP " + str(gp_index + 1) + ":")
            gp.print_model()

    def train_gp(self, gp_index, optimization_opt_dict):
        """
        Call train_gp_likelihood
        """
        self.train_gp_likelihood(gp_index, optimization_opt_dict)
        if self.approximation_mode == "SOR":
            print("\nSelect the SOR regressors...")
            with torch.no_grad():
                # permutation_indices = np.random.permutation(self.gp_inputs.shape[0])
                permutation_indices = np.arange(0, self.gp_inputs.shape[0])
                self.reg_indices_SOR_list[gp_index] = self.gp_list[gp_index].set_inducing_inputs_from_data(
                    X=self.gp_inputs[permutation_indices, :],
                    Y=self.gp_output_list[gp_index][permutation_indices, :],
                    threshold=self.approximation_dict["threshold"][gp_index],
                    flg_regressors_trainable=self.approximation_dict["flg_regressors_trainable"],
                )

    def train_gp_likelihood(self, gp_index, optimization_opt_dict):
        """
        Train the gp with index gp_index optimizing the likelihood
        """
        # check the batch size
        batch_size = self.gp_inputs.shape[0]

        # get the dataloader
        if self.flg_norm:
            self.norm_list[gp_index] = torch.max(torch.abs(self.gp_output_list[gp_index]))
        dataset = torch.utils.data.TensorDataset(
            self.gp_inputs, self.gp_output_list[gp_index] / self.norm_list[gp_index]
        )
        # dataset = torch.utils.data.TensorDataset(self.gp_inputs, self.gp_output_list[gp_index])
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # fit the model
        f_optim = eval(optimization_opt_dict["f_optimizer"])
        self.gp_list[gp_index].fit_model(
            trainloader=trainloader,
            optimizer=f_optim(self.gp_list[gp_index].parameters()),
            criterion=optimization_opt_dict["criterion"](),
            N_epoch=optimization_opt_dict["N_epoch"],
            N_epoch_print=optimization_opt_dict["N_epoch_print"],
        )

    def train_SOR_gp_likelihood(self, gp_index, optimization_opt_dict):
        """
        Train the gp with index gp_index optimizing the likelihood
        """
        # check the batch size
        batch_size = self.gp_inputs.shape[0]

        # get the dataloader
        dataset = torch.utils.data.TensorDataset(self.gp_inputs, self.gp_output_list[gp_index])
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # fit the model
        f_optim = eval(optimization_opt_dict["f_optimizer"])
        self.gp_list[gp_index].fit_SOR_model(
            trainloader=trainloader,
            optimizer=f_optim(self.gp_list[gp_index].parameters()),
            criterion=optimization_opt_dict["criterion"](),
            N_epoch=optimization_opt_dict["N_epoch"],
            N_epoch_print=optimization_opt_dict["N_epoch_print"],
        )

    def get_gp(self, gp_index, init_dict):
        """
        Returns the num_index gp
        """
        raise NotImplementedError()
        return None

    def data_to_gp_input(self, states, inputs):
        """
        Returns gp input given states and inputs.
        In this basic implementation gp inputs are the concatenation
        of states and inputs
        """
        return torch.cat([states, inputs], 1)

    def data_to_gp_output(self, states):
        """
        Returns a list with the gp ouputs given the states.
        In this basic implementation gp output are the delta in each state dimension
        """
        return [(states[1:, i] - states[:-1, i]).reshape([-1, 1]) for i in range(0, self.dim_state)]

    def data_to_gp_IO(self, states, inputs, exogenous_inputs=None):
        """
        Returns the GP dataset given states and inputs
        """
        return (
            self._data_to_gp_input_with_optional_exogenous(states, inputs, exogenous_inputs)[:-1, :],
            self.data_to_gp_output(states),
        )

    def get_next_state_from_gp_output(
        self, current_state, current_input, gp_output_mean_list, gp_output_var_list, particle_pred=True
    ):
        """
        Returns next state samples with mean and variance of GP outputs, given:
        -the current state
        -the current inputs
        -a list with mean and variance of the gp output
        """
        #  mean and variance of delta distribution
        delta_mean = torch.cat(gp_output_mean_list, 1)
        delta_var = torch.cat(gp_output_var_list, 1)
        if particle_pred == True:
            # sample delta from distribution
            delta_distribution = Normal(delta_mean, torch.sqrt(torch.clamp(delta_var, min=1e-8)))
            delta_sample = delta_distribution.rsample()
            # delta_sample = delta_mean + torch.sqrt(delta_var)*torch.randn(delta_mean.shape, dtype=self.dtype, device=self.device)
        else:
            delta_sample = delta_mean
        # get the next state
        next_states = current_state + delta_sample
        # return the next state and the delta distribution
        return next_states, delta_mean, delta_var


class Model_learning_RBF(Model_learning):
    """
    Model learning class with all the gp given by RBF kernel
    """

    def __init__(
        self,
        num_gp,
        init_dict_list,
        approximation_mode=None,
        approximation_dict=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        flg_norm=False,
    ):
        super(Model_learning_RBF, self).__init__(
            num_gp=num_gp,
            init_dict_list=init_dict_list,
            approximation_mode=approximation_mode,
            approximation_dict=approximation_dict,
            dtype=dtype,
            device=device,
            flg_norm=flg_norm,
        )

    def get_gp(self, gp_index, init_dict):
        """
        Returns the num_index gp
        """
        return SGP.RBF(**init_dict)


class Model_learning_RBF_angle_state(Model_learning):
    """
    Model learning class with all the gp given by RBF kernel and
    the possibility to use sin-cos angle representation in GP inputs
    """

    def __init__(
        self,
        num_gp,
        init_dict_list,
        angle_indeces,
        not_angle_indeces,
        approximation_mode=None,
        approximation_dict=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        flg_norm=False,
    ):
        super(Model_learning_RBF_angle_state, self).__init__(
            num_gp=num_gp,
            init_dict_list=init_dict_list,
            approximation_mode=approximation_mode,
            approximation_dict=approximation_dict,
            dtype=dtype,
            device=device,
            flg_norm=flg_norm,
        )
        self.angle_indeces = angle_indeces
        self.not_angle_indeces = not_angle_indeces

    def get_gp(self, gp_index, init_dict):
        """
        Returns the num_index gp
        """
        return SGP.RBF(**init_dict)

    def data_to_gp_input(self, states, inputs):
        """
        Returns gp input given states and inputs.
        Extended state is considered: [x, x_dot, theta_dot, sin(theta), cos(theta)]
        where state = [x, theta], with theta angular components of the state and x the other states
        """
        extended_states = torch.cat(
            [
                states[:, self.not_angle_indeces],
                torch.sin(states[:, self.angle_indeces]),
                torch.cos(states[:, self.angle_indeces]),
            ],
            1,
        )

        return torch.cat([extended_states, inputs], 1)


class Model_learning_RBF_angle_state_with_exogenous(Model_learning_RBF_angle_state):
    """
    RBF model learning with known exogenous inputs appended to the GP input.
    The control policy still sees only the measured state; exogenous inputs are
    used only by the learned dynamics model.
    """

    def __init__(
        self,
        num_gp,
        init_dict_list,
        angle_indeces,
        not_angle_indeces,
        approximation_mode=None,
        approximation_dict=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        flg_norm=False,
    ):
        super(Model_learning_RBF_angle_state_with_exogenous, self).__init__(
            num_gp=num_gp,
            init_dict_list=init_dict_list,
            angle_indeces=angle_indeces,
            not_angle_indeces=not_angle_indeces,
            approximation_mode=approximation_mode,
            approximation_dict=approximation_dict,
            dtype=dtype,
            device=device,
            flg_norm=flg_norm,
        )
        self.use_exogenous_inputs = True

    def data_to_gp_input(self, states, inputs, exogenous_inputs=None):
        if exogenous_inputs is None:
            raise ValueError("exogenous_inputs must be provided for road-aware GP model learning.")
        base_inputs = super(Model_learning_RBF_angle_state_with_exogenous, self).data_to_gp_input(states, inputs)
        return torch.cat([base_inputs, exogenous_inputs], 1)


class Model_learning_RBF_MPK_angle_state(Model_learning_RBF_angle_state):
    """
    Model learning class with all the gp given by a combination of RBF and MP kernel and
    the possibility to use sin-cos angle representation in GP inputs
    """

    def __init__(
        self,
        num_gp,
        init_dict_list,
        angle_indeces,
        not_angle_indeces,
        approximation_mode=None,
        approximation_dict=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        flg_norm=False,
    ):
        super(Model_learning_RBF_MPK_angle_state, self).__init__(
            num_gp=num_gp,
            init_dict_list=init_dict_list,
            angle_indeces=angle_indeces,
            not_angle_indeces=not_angle_indeces,
            approximation_mode=approximation_mode,
            approximation_dict=approximation_dict,
            dtype=dtype,
            device=device,
            flg_norm=flg_norm,
        )

    def get_gp(self, gp_index, init_dict):
        gp_list = []
        gp_list.append(SGP.RBF(**init_dict[0]))
        gp_list.append(Sparse_GP.get_Volterra_MPK_GP(**init_dict[1]))
        return GP.Sum_Independent_GP(*gp_list)


class Speed_Model_learning_RBF_angle_state(Model_learning):
    """
    Speed model learning class with all the gp given by RBF kernel.
    The GP model predicts speed changes. vel_indeces and not_vel_indeces are related:
    1st-index-state in vel_indeces is the derivative of 1st-index-state in not_vel_indeces.
    """

    def __init__(
        self,
        num_gp,
        init_dict_list,
        T_sampling,
        angle_indeces,
        not_angle_indeces,
        vel_indeces,
        not_vel_indeces,
        approximation_mode=None,
        approximation_dict=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        flg_norm=False,
    ):
        super(Speed_Model_learning_RBF_angle_state, self).__init__(
            num_gp=num_gp,
            init_dict_list=init_dict_list,
            approximation_mode=approximation_mode,
            approximation_dict=approximation_dict,
            dtype=dtype,
            device=device,
            flg_norm=flg_norm,
        )
        self.vel_indeces = vel_indeces
        self.not_vel_indeces = not_vel_indeces
        self.angle_indeces = angle_indeces
        self.not_angle_indeces = not_angle_indeces
        self.T_sampling = T_sampling

    def get_gp(self, gp_index, init_dict):
        """
        Returns the num_index gp
        """
        return SGP.RBF(**init_dict)

    def data_to_gp_output(self, states):
        """
        Returns a list with the gp ouputs given the states.
        GP outputs are the speed changes.
        """

        return [(states[1:, i] - states[:-1, i]).reshape([-1, 1]) for i in self.vel_indeces]

    def data_to_gp_input(self, states, inputs):
        """
        Returns gp input given states and inputs.
        sin-cos extended state is the gp input vector.
        """
        extended_states = torch.cat(
            [
                states[:, self.not_angle_indeces],
                torch.sin(states[:, self.angle_indeces]),
                torch.cos(states[:, self.angle_indeces]),
            ],
            1,
        )
        return torch.cat([extended_states, inputs], 1)

    def get_next_state_from_gp_output(
        self, current_state, current_input, gp_output_mean_list, gp_output_var_list, particle_pred=True
    ):
        """
        Returns next state samples with mean and variance of GP outputs, given:
        -the current state
        -the current inputs
        -a list with mean and variance of the gp output
        by integrating the speed changes (GP outputs)
        """
        #  mean and variance of delta speed distribution
        delta_vel_mean = torch.cat(gp_output_mean_list, 1)
        delta_vel_var = torch.cat(gp_output_var_list, 1)

        # preallocate variables
        next_states = torch.zeros(current_state.shape, dtype=self.dtype, device=self.device)

        if particle_pred == True:
            # sample delta speed from distribution
            delta_speed_distribution = Normal(delta_vel_mean, torch.sqrt(torch.clamp(delta_vel_var, min=1e-8)))
            delta_speed_sample = delta_speed_distribution.rsample()
            # delta_speed_sample = delta_vel_mean + torch.sqrt(delta_vel_var)*torch.randn(delta_vel_mean.shape, dtype=self.dtype, device=self.device)
        else:
            delta_speed_sample = delta_vel_mean

        # compute next states
        next_states[:, self.vel_indeces] = current_state[:, self.vel_indeces] + delta_speed_sample
        next_states[:, self.not_vel_indeces] = (
            current_state[:, self.not_vel_indeces]
            + self.T_sampling * current_state[:, self.vel_indeces]
            + self.T_sampling / 2 * delta_speed_sample
        )

        return next_states, delta_vel_mean, delta_vel_var


class Speed_Model_learning_RBF_MPK_angle_state(Speed_Model_learning_RBF_angle_state):
    """
    Models the velocity of each delta of velocity with a RBF+MPK
    """

    def __init__(
        self,
        num_gp,
        init_dict_list,
        T_sampling,
        angle_indeces,
        not_angle_indeces,
        vel_indeces,
        not_vel_indeces,
        approximation_mode=None,
        approximation_dict=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        flg_norm=False,
    ):
        super(Speed_Model_learning_RBF_MPK_angle_state, self).__init__(
            num_gp=num_gp,
            init_dict_list=init_dict_list,
            T_sampling=T_sampling,
            angle_indeces=angle_indeces,
            not_angle_indeces=not_angle_indeces,
            vel_indeces=vel_indeces,
            not_vel_indeces=not_vel_indeces,
            approximation_mode=approximation_mode,
            approximation_dict=approximation_dict,
            dtype=dtype,
            device=device,
            flg_norm=flg_norm,
        )

    def get_gp(self, gp_index, init_dict):
        gp_list = []
        gp_list.append(SGP.RBF(**init_dict[0]))
        gp_list.append(Sparse_GP.get_Volterra_MPK_GP(**init_dict[1]))
        return GP.Sum_Independent_GP(*gp_list)


class SP_Speed_Model_learning_Furuta(Model_learning):
    """
    Speed model learning class for the FP with semiparametric kernel.
    The GP model predicts speed changes. vel_indeces and not_vel_indeces are related:
    1st-index-state in vel_indeces is the derivative of 1st-index-state in not_vel_indeces.
    The state is assumed to be:  [theta_hor, theta_ver, theta_hor_dot, theta_ver_dot]
    """

    def __init__(
        self,
        num_gp,
        init_dict_list,
        T_sampling,
        vel_indeces,
        not_vel_indeces,
        approximation_mode=None,
        approximation_dict=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        flg_norm=False,
    ):
        super(SP_Speed_Model_learning_Furuta, self).__init__(
            num_gp=num_gp,
            init_dict_list=init_dict_list,
            approximation_mode=approximation_mode,
            approximation_dict=approximation_dict,
            dtype=dtype,
            device=device,
            flg_norm=flg_norm,
        )
        self.vel_indeces = vel_indeces
        self.not_vel_indeces = not_vel_indeces
        self.T_sampling = T_sampling

    def get_gp(self, gp_index, init_dict):
        """
        Returns the num_index gp
        """
        gp_list = []
        # get the RBF GP
        gp_list.append(SGP.RBF(**init_dict[0]))
        # get the model-based GP
        gp_list.append(Sparse_GP.Linear_GP(**init_dict[1]))
        # return the SP GP
        return GP.Sum_Independent_GP(*gp_list)

    def data_to_gp_output(self, states):
        """
        Returns a list with the gp ouputs given the states.
        GP outputs are the speed changes.
        """

        return [(states[1:, i] - states[:-1, i]).reshape([-1, 1]) for i in self.vel_indeces]

    def data_to_gp_input(self, states, inputs):
        """
        Returns gp input given states and inputs.
        extended state is the gp input vector, which accounts for the FP state and
        some features suggested by the forward dynamics model.
        """
        extended_states = torch.cat(
            [
                states,
                inputs,
                torch.sin(states[:, 1:2]) * states[:, 3:4] ** 2,
                states[:, 2:3] * states[:, 3:4] * torch.sin(2 * states[:, 1:2]),
                states[:, 2:3],
                states[:, 2:3] ** 2 * torch.sin(2 * states[:, 1:2]),
                states[:, 3:4],
                torch.sin(states[:, 1:2]),
                inputs * torch.cos(states[:, 1:2]),
            ],
            1,
        )
        return extended_states

    def get_next_state_from_gp_output(
        self, current_state, current_input, gp_output_mean_list, gp_output_var_list, particle_pred=True
    ):
        """
        Returns next state samples with mean and variance of GP outputs, given:
        -the current state
        -the current inputs
        -a list with mean and variance of the gp output
        by integrating the speed changes (GP outputs)
        """
        #  mean and variance of delta speed distribution
        delta_vel_mean = torch.cat(gp_output_mean_list, 1)
        delta_vel_var = torch.cat(gp_output_var_list, 1)

        # preallocate variables
        next_states = torch.zeros(current_state.shape, dtype=self.dtype, device=self.device)

        if particle_pred == True:
            # sample delta speed from distribution
            delta_speed_distribution = Normal(delta_vel_mean, torch.sqrt(torch.clamp(delta_vel_var, min=1e-8)))
            delta_speed_sample = delta_speed_distribution.rsample()
            # delta_speed_sample = delta_vel_mean + torch.sqrt(delta_vel_var)*torch.randn(delta_vel_mean.shape, dtype=self.dtype, device=self.device)
        else:
            delta_speed_sample = delta_vel_mean

        # compute next states
        next_states[:, self.vel_indeces] = current_state[:, self.vel_indeces] + delta_speed_sample
        next_states[:, self.not_vel_indeces] = (
            current_state[:, self.not_vel_indeces]
            + self.T_sampling * current_state[:, self.vel_indeces]
            + self.T_sampling / 2 * delta_speed_sample
        )

        return next_states, delta_vel_mean, delta_vel_var


class Model_learning_Quarter_Car_Gym_State_Reconstruction(Model_learning_RBF_angle_state):
    """
    Model learning class for Gym Quarter Car that reconstructs states.
    GP inputs: [suspension_deflection, v_def, vs, vu, u]
    GP outputs: [acc_s_change, vs_change, suspension_deflection_change, v_def_change]
    """

    def __init__(
        self,
        num_gp,
        init_dict_list,
        angle_indeces,
        not_angle_indeces,
        obs_scaling,
        approximation_mode=None,
        approximation_dict=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        flg_norm=False,
        use_road_gp_input=False,
    ):
        super(Model_learning_Quarter_Car_Gym_State_Reconstruction, self).__init__(
            num_gp=num_gp,
            init_dict_list=init_dict_list,
            angle_indeces=angle_indeces,
            not_angle_indeces=not_angle_indeces,
            approximation_mode=approximation_mode,
            approximation_dict=approximation_dict,
            dtype=dtype,
            device=device,
            flg_norm=flg_norm,
        )
        self.obs_scaling = torch.tensor(obs_scaling, dtype=self.dtype, device=self.device)
        self.use_exogenous_inputs = use_road_gp_input

    def to(self, device):
        super(Model_learning_Quarter_Car_Gym_State_Reconstruction, self).to(device)
        self.obs_scaling = self.obs_scaling.to(device)

    def data_to_gp_input(self, states, inputs, exogenous_inputs=None):
        """
        unscales observations and reconstructs physical wheel velocity:
        v_u = v_s - v_def
        """
        unscaled_states = states * self.obs_scaling
        if states.shape[1] == 5:
            v_s = unscaled_states[:, 2:3]  # sprung velocity
            susp_def = unscaled_states[:, 3:4]  # suspension deflection
            v_def = unscaled_states[:, 4:5]  # suspension deflection velocity
        else:
            v_s = unscaled_states[:, 1:2]  # sprung velocity
            susp_def = unscaled_states[:, 2:3]  # suspension deflection
            v_def = unscaled_states[:, 3:4]  # suspension deflection velocity

        # Reconstruct unsprung velocity
        v_u = v_s - v_def

        # Reconstruct physical coordinate state vector
        reconstructed_states = torch.cat([susp_def, v_def, v_s, v_u], dim=1)

        gp_in = torch.cat([reconstructed_states, inputs], dim=1)
        if self.use_exogenous_inputs:
            if exogenous_inputs is None:
                raise ValueError("exogenous_inputs must be provided for road-aware GP model learning.")
            gp_in = torch.cat([gp_in, exogenous_inputs], dim=1)
        return gp_in


class Model_learning_Quarter_Car_Gym_Physics_Residual(Model_learning_RBF_angle_state):
    """
    Model learning class for Gym Quarter Car that does physics-informed residual learning.
    GP inputs: [suspension_deflection, v_def, vs, vu, u] (reconstructed physical coordinate space)
    GP targets: delta_obs_observed - delta_obs_physics
    """

    def __init__(
        self,
        num_gp,
        init_dict_list,
        angle_indeces,
        not_angle_indeces,
        obs_scaling,
        act_scaling,
        T_sampling,
        m_s,
        m_u,
        k_s,
        c_s,
        k_t,
        c_t=0.0,
        approximation_mode=None,
        approximation_dict=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        flg_norm=False,
        use_road_gp_input=False,
    ):
        super(Model_learning_Quarter_Car_Gym_Physics_Residual, self).__init__(
            num_gp=num_gp,
            init_dict_list=init_dict_list,
            angle_indeces=angle_indeces,
            not_angle_indeces=not_angle_indeces,
            approximation_mode=approximation_mode,
            approximation_dict=approximation_dict,
            dtype=dtype,
            device=device,
            flg_norm=flg_norm,
        )
        self.obs_scaling = torch.tensor(obs_scaling, dtype=self.dtype, device=self.device)
        self.use_exogenous_inputs = use_road_gp_input
        self.act_scaling = act_scaling
        self.T_sampling = T_sampling
        self.m_s = m_s
        self.m_u = m_u
        self.k_s = k_s
        self.c_s = c_s
        self.k_t = k_t
        self.c_t = c_t

    def to(self, device):
        super(Model_learning_Quarter_Car_Gym_Physics_Residual, self).to(device)
        self.obs_scaling = self.obs_scaling.to(device)

    def data_to_gp_input(self, states, inputs, exogenous_inputs=None):
        """
        unscales observations and reconstructs physical wheel velocity:
        v_u = v_s - v_def
        """
        unscaled_states = states * self.obs_scaling
        if states.shape[1] == 5:
            v_s = unscaled_states[:, 2:3]
            susp_def = unscaled_states[:, 3:4]
            v_def = unscaled_states[:, 4:5]
        else:
            v_s = unscaled_states[:, 1:2]
            susp_def = unscaled_states[:, 2:3]
            v_def = unscaled_states[:, 3:4]
        v_u = v_s - v_def
        reconstructed_states = torch.cat([susp_def, v_def, v_s, v_u], dim=1)
        gp_in = torch.cat([reconstructed_states, inputs], dim=1)
        if self.use_exogenous_inputs:
            if exogenous_inputs is None:
                raise ValueError("exogenous_inputs must be provided for road-aware GP model learning.")
            gp_in = torch.cat([gp_in, exogenous_inputs], dim=1)
        return gp_in

    def get_physics_delta_obs(self, states, inputs):
        """
        Calculates nominal physical observation changes in scaled observation space over T_sampling
        (Assuming z_r = 0, z_r_dot = 0)
        """
        unscaled_states = states * self.obs_scaling
        if states.shape[1] == 5:
            acc_s = unscaled_states[:, 0:1]  # physical sprung mass acceleration
            x_s = unscaled_states[:, 1:2]  # physical sprung mass position
            v_s = unscaled_states[:, 2:3]  # physical sprung mass velocity
            susp_def = unscaled_states[:, 3:4]  # physical suspension deflection (z_s - z_u)
            v_def = unscaled_states[:, 4:5]  # physical deflection velocity (z_s_dot - z_u_dot)
        else:
            acc_s = unscaled_states[:, 0:1]  # physical sprung mass acceleration
            v_s = unscaled_states[:, 1:2]  # physical sprung mass velocity
            susp_def = unscaled_states[:, 2:3]  # physical suspension deflection (z_s - z_u)
            v_def = unscaled_states[:, 3:4]  # physical deflection velocity (z_s_dot - z_u_dot)

        # Reconstruct physical wheel velocity
        v_u = v_s - v_def
        # Reconstruct physical wheel displacement (approximating z_s ≈ 0, so z_u ≈ -susp_def)
        z_u = -susp_def

        # Physical force from action (Gym action is scaled, physical force = action / act_scaling)
        u_phys = inputs[:, 0:1] / self.act_scaling

        # Calculate nominal physical suspension force
        F_susp = self.k_s * susp_def + self.c_s * v_def
        # Calculate nominal physical tire force (assuming z_r = 0, z_r_dot = 0)
        # We assume F_tire = 0 as a baseline since tire deflection is tiny and oscillates around 0.
        # This prevents massive fictitious forces when z_u is unknown.
        F_tire = torch.zeros_like(susp_def)  # hat!!

        # Calculate nominal sprung and unsprung accelerations
        z_s_ddot = (-F_susp + u_phys) / self.m_s
        z_u_ddot = (F_susp - F_tire - u_phys) / self.m_u

        # Calculate nominal changes over T_sampling in physical coordinates
        delta_v_s = acc_s * self.T_sampling
        delta_susp_def = v_def * self.T_sampling
        delta_v_def = (acc_s - z_u_ddot) * self.T_sampling
        delta_acc_s = torch.zeros_like(acc_s)

        # Combine nominal physical changes
        if states.shape[1] == 5:
            delta_x_s = v_s * self.T_sampling
            delta_phys_unscaled = torch.cat([delta_acc_s, delta_x_s, delta_v_s, delta_susp_def, delta_v_def], dim=1)
            return delta_phys_unscaled / self.obs_scaling
        else:
            delta_phys_unscaled = torch.cat([delta_acc_s, delta_v_s, delta_susp_def, delta_v_def], dim=1)
            return delta_phys_unscaled / self.obs_scaling

    def data_to_gp_IO(self, states, inputs, exogenous_inputs=None):
        """
        GP targets: observed scaled delta - nominal physics scaled delta
        """
        gp_inputs = self.data_to_gp_input(states, inputs, exogenous_inputs=exogenous_inputs)[:-1, :]

        # Observed differences in scaled observation space
        observed_delta = states[1:] - states[:-1]

        # Nominal physics differences in scaled observation space
        physics_delta = self.get_physics_delta_obs(states[:-1], inputs[:-1])

        # Residuals to be learned by GP
        residuals = observed_delta - physics_delta

        # Split into list of outputs for each GP
        gp_output_list = [residuals[:, i : i + 1] for i in range(self.num_gp)]

        return gp_inputs, gp_output_list

    def get_next_state_from_gp_output(
        self, current_state, current_input, gp_output_mean_list, gp_output_var_list, particle_pred=True
    ):
        """
        Reconstructs the next observation by adding GP residual predictions back to nominal physical updates
        """
        # GP predicted residual change
        residual_mean = torch.cat(gp_output_mean_list, dim=1)
        residual_var = torch.cat(gp_output_var_list, dim=1)

        # Nominal physics change in scaled space
        physics_delta = self.get_physics_delta_obs(current_state, current_input)

        # Total observation change in scaled space
        delta_obs_mean = physics_delta + residual_mean

        if particle_pred == True:
            delta_obs_distribution = Normal(delta_obs_mean, torch.sqrt(torch.clamp(residual_var, min=1e-8)))
            delta_obs_sample = delta_obs_distribution.rsample()
        else:
            delta_obs_sample = delta_obs_mean

        next_states = current_state + delta_obs_sample
        return next_states, delta_obs_mean, residual_var
