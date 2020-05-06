###for ease if later want to add acq
import numpy as np
import math
from typing import Dict, Optional, Tuple, Union
import botorch,torch
from botorch.utils.transforms import t_batch_mode_transform

#Bay Opt currently will only chose the maximum of the acqusition function returns; ensure these are maximums desired

def Acq(gp,cur_min,a_type='EI'):
    if a_type=='EI':
        return noised_EI(model=gp, best_f=cur_min,maximize=False)
    elif a_type=='SPE':
        return SquaredPosteriorError(model=gp)
    else:
        raise ValueError('not a valid acquisition function')


class noised_EI(botorch.acquisition.analytic.AnalyticAcquisitionFunction):
    r"""Single-outcome Expected Improvement (analytic) with observation noise

    Computes classic Expected Improvement over the current best observed value,
    using the analytic formula for a Normal posterior distribution. Unlike the
    MC-based acquisition functions, this relies on the posterior at single test
    point being Gaussian (and require the posterior to implement `mean` and
    `variance` properties). Only supports the case of `q=1`. The model must be
    single-outcome.

    `EI(x) = E(max(y - best_f, 0)), y ~ f(x)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EI = ExpectedImprovement(model, best_f=0.2)
        >>> ei = EI(test_X)
    """

    def __init__(
        self, model: botorch.models.model.Model, best_f: Union[float, torch.Tensor], maximize: bool = True
    ) -> None:
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `b1 x ... bk x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `b1 x ... bk`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        self.best_f = self.best_f.to(X)
        posterior = self.model.posterior(X,observation_noise=True)
        self._validate_single_output_posterior(posterior)
        mean = posterior.mean
        # deal with batch evaluation and broadcasting
        view_shape = mean.shape[:-2] if mean.dim() >= X.dim() else X.shape[:-2]
        mean = mean.view(view_shape)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().view(view_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        return ei


class SquaredPosteriorError(botorch.acquisition.AnalyticAcquisitionFunction):
    r"""Single-outcome Posterior Variance.

    Only supports the case of q=1. Requires the model's posterior to have a
    `variance` property. The model must be single-outcome.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> PM = PosteriorMean(model)
        >>> pm = PM(test_X)
    """

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the posterior variance on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Posterior variance values at the given design
            points `X`.
        """
        posterior = self._get_posterior(X=X)
        return posterior.variance.clamp_min(1e-9).view(X.shape[:-2])

