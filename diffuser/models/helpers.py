import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb

import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

#-----------------------------------------------------------------------------#
#--------------------------------- attention ---------------------------------#
#-----------------------------------------------------------------------------#

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)

#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x

def apply_direction_coord_conditioning(x, conditions, action_dim):
    # Takes as condition the angle and head coordinates only.
    for t, val in conditions.items():
        x[:, t, -3:] = torch.unsqueeze(val[:, -3:], axis=0).clone()
    return x

def apply_direction_coord_init_conditioning(x, conditions, action_dim):
    # Takes as condition the angle and head coordinates only.
    for t, val in conditions.items():
        x[:, t, -3:] = torch.unsqueeze(val[:, -3:], axis=0).clone()
        x[:, t, -6] = torch.unsqueeze(val[:, -6], axis=0).clone()
    return x

def apply_simplified_direction_coord_init_conditioning(x, conditions, action_dim):
    # Takes as condition the angle and head coordinates only.
    for t, val in conditions.items():
        x[:, t, -3:] = torch.unsqueeze(val[:, -3:], axis=0).clone()
    return x

def apply_direction_coord_end_conditioning(x, conditions, action_dim):
    # Takes as condition the angle and head coordinates only.
    for t, val in conditions.items():
        x[:, t, -3:] = torch.unsqueeze(val[:, -3:], axis=0).clone()
        x[:, t, -6] = torch.unsqueeze(val[:, -6], axis=0).clone()
        x[:, t, -10] = torch.unsqueeze(val[:, -10], axis=0).clone()

    return x

def apply_direction_head_conditioning(x, conditions, action_dim):
    # Takes as condition the angle and head coordinates only.
    for t, val in conditions.items():
        x[:, t, -3:] = torch.unsqueeze(val[:, -3:], axis=0).clone()
        x[:, t, -6] = torch.unsqueeze(val[:, -6], axis=0).clone()
    return x

def apply_harder_direction_coord_conditioning(x, conditions, action_dim):
    # Takes as condition the angle and head coordinates only.
    for t, val in conditions.items():
        x[:, t, -5:] = torch.unsqueeze(val[:, -5:], axis=0).clone()
        x[:, t, action_dim] = torch.unsqueeze(val[:, 40], axis=0).clone()
        x[:, t, action_dim+39] = torch.unsqueeze(val[:, 49], axis=0).clone()
        x[:, t, action_dim+40] = torch.unsqueeze(val[:, 40], axis=0).clone()
        x[:, t, action_dim+49] = torch.unsqueeze(val[:, 49], axis=0).clone()

    return x

def apply_direction_restriction_noisy_cond(x, conditions, action_dim):
    # Takes as condition the angle and head coordinates only.
    for t, val in conditions.items():
        x[:, t, action_dim:-3] = torch.unsqueeze(val[:, :-3], axis=0).clone()
        x[:, t, -2:] = torch.unsqueeze(val[:, -2:], axis=0).clone()

    return x

#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = ((loss[:, 0, :self.action_dim] + 1e-10) / (self.weights[0, :self.action_dim]+ 1e-10)).mean()
        return weighted_loss, {'a0_loss': a0_loss}

class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(),
                utils.to_np(targ).squeeze()
            )[0,1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class SpermLoss(WeightedLoss):

    def _loss(self, pred, targ):
        mask = torch.ones_like(pred) * 0.25
        mask[:, :, -5:-2] = 1.  # give more importance to the head angle
        mask[:, :, -10:-5] = 1  # give more importance to the head displacement

        return F.mse_loss(pred, targ, reduction='none') * mask

class SpermLossEMA(WeightedLoss):

    def _ema(self, pred):
        alpha = 0.25
        ema = torch.tensor(np.zeros((pred.shape[0], pred.shape[1])), device=pred.device, dtype=torch.float32)

        ema[:, 0] = pred[:, 0, -3]

        for i in range(1, pred.shape[1]):
            ema[:, i] = (alpha * pred[:, i, -3] + (1 - alpha) * ema[:, i-1])

        return ema

    def _loss(self, pred, targ):
        mask = torch.ones_like(pred) * 0.1
        mask[:, :, -3] = 0.  # give more importance to the head angle
        mask[:, :, -5:-3] = 1.  # give more importance to the head displacement

        ema = self._ema(pred)
        return F.mse_loss(pred, targ, reduction='none') * mask, F.mse_loss(ema, targ[:, :, 3], reduction='none')

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss, ema_loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = ((loss[:, 0, :self.action_dim] + 1e-10) / (self.weights[0, :self.action_dim]+ 1e-10)).mean()
        return weighted_loss + torch.mean(ema_loss), {'a0_loss': a0_loss}

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
    'sperm_loss': SpermLoss,
    'sperm_ema_loss': SpermLossEMA
}
