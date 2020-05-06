#! /usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
Gaussian Process Regression models based on GPyTorch models.
"""

from typing import Any, Optional

import torch
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    _GaussianLikelihoodBase,
)
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.module import Module
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor

from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.utils import validate_input_scaling
import Mean_Function

MIN_INFERRED_NOISE_LEVEL = 1e-4


class SingleTaskGP(BatchedMultiOutputGPyTorchModel, ExactGP):
    r"""A single-task exact GP model.

    A single-task exact GP using relatively strong priors on the Kernel
    hyperparameters, which work best when covariates are normalized to the unit
    cube and outcomes are standardized (zero mean, unit variance).

    This model works in batch mode (each batch having its own hyperparameters).
    When the training observations include multiple outputs, this model will use
    batching to model outputs independently.

    Use this model when you have independent output(s) and all outputs use the same
    training data. If outputs are independent and outputs have different training
    data, use the ModelListGP. When modeling correlations between outputs, use
    the MultiTaskGP.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        stats_X: Any,
        stats_Y: Any,
        likelihood: Optional[Likelihood] = None,
        covar_module: Optional[Module] = None,        
        mean_module: Optional[Module] = None,
    ) -> None:
        r"""A single-task exact GP model.

        Args:
            train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
                training observations.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
            covar_module: The covariance (kernel) matrix. If omitted, use the
                MaternKernel.

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
            >>> model = SingleTaskGP(train_X, train_Y)
        """
        validate_input_scaling(train_X=train_X, train_Y=train_Y)
        self._validate_tensor_args(X=train_X, Y=train_Y)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        if likelihood is None:
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                batch_shape=self._aug_batch_shape,
                noise_constraint=GreaterThan(
                    MIN_INFERRED_NOISE_LEVEL,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )
        else:
            self._is_custom_likelihood = True
        ExactGP.__init__(self, train_X, train_Y, likelihood)
        if mean_module is None:
            self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        else:
            self.mean_module = Mean_Function.Ed_Mean(mean_module,stats_X,stats_Y,batch_shape=self._aug_batch_shape)
        if covar_module is None:
            self.covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=train_X.shape[-1],
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
        else:
            self.covar_module = covar_module
        self.to(train_X)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
