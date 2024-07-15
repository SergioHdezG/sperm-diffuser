from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    apply_direction_coord_conditioning,
    apply_harder_direction_coord_conditioning,
    apply_direction_restriction_noisy_cond,
    Losses,
)


Sample = namedtuple('Sample', 'trajectories values chains')


@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)
        self.apply_conditioning = apply_conditioning

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = self.apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = self.apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = self.apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = self.apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)

class GaussianDiffusionImitation(GaussianDiffusion):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps=n_timesteps,
        loss_type=loss_type, clip_denoised=clip_denoised, predict_epsilon=predict_epsilon,
        action_weight=action_weight, loss_discount=loss_discount, loss_weights=loss_weights)

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = self.apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = self.apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)


class GaussianDiffusionImitationCondition(GaussianDiffusionImitation):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
        fn_condition=apply_harder_direction_coord_conditioning):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps=n_timesteps,
        loss_type=loss_type, clip_denoised=clip_denoised, predict_epsilon=predict_epsilon,
        action_weight=action_weight, loss_discount=loss_discount, loss_weights=loss_weights)

        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)
        self.apply_conditioning = fn_condition


    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = self.apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = self.apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        # x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[:, :self.action_dim] = action_weight

        return loss_weights

class GaussianDiffusionImitationNoisyCondition(GaussianDiffusionImitationCondition):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
        fn_condition=apply_direction_restriction_noisy_cond, cond_portion=0.75):
        super().__init__(model=model, horizon=horizon, observation_dim=observation_dim, action_dim=action_dim,
                         n_timesteps=n_timesteps, loss_type=loss_type, clip_denoised=clip_denoised,
                         predict_epsilon=predict_epsilon, action_weight=action_weight, loss_discount=loss_discount,
                         loss_weights=loss_weights, fn_condition=fn_condition)
        self.cond_portion = cond_portion

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        aux_cond = self.q_sample_conditions(cond, torch.tensor(np.asarray([self.n_timesteps], dtype=np.int)).to(x.device), x, batch_size=x.shape[0])
        aux_init_cond = self.q_sample_conditions(cond, torch.tensor(np.asarray([self.n_timesteps], dtype=np.int)).to(x.device), x, batch_size=x.shape[0], half_cond=True)

        x = self.apply_conditioning(x, aux_cond, self.action_dim)

        chain = [x] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()

        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)

            if torch.all(t >= int(self.n_timesteps * self.cond_portion)):
                x = self.apply_conditioning(x, aux_cond, self.action_dim)
            else:
                x = self.apply_conditioning(x, aux_init_cond, self.action_dim)


            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    def q_sample_conditions(self, cond, t, noise, batch_size, half_cond=False):
        aux_cond = {}
        # Apply noise to conditioning
        noise_t = torch.from_numpy(
            np.asarray([int(self.n_timesteps * self.cond_portion) for _ in range(batch_size)], dtype=np.int)).to(
            noise.device)
        noise_t = torch.minimum(noise_t, t)
        for key in cond.keys():
            if key == 0:
                aux_cond[key] = cond[key]
            elif not half_cond:
                aux_cond[key] = self.q_sample(x_start=cond[key].to(noise.device), t=noise_t, noise=noise[:, key, self.action_dim:])
        return aux_cond

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        cond = self.q_sample_conditions(cond, t, noise, batch_size=x_start.shape[0])

        x_noisy = self.apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = self.apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

class ValueDiffusion(GaussianDiffusion):

    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = self.apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t):
        return self.model(x, cond, t)

