#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

from dataclasses import dataclass
from disent.frameworks.vae.unsupervised import Vae


# ========================================================================= #
# Beta-VAE Loss                                                             #
# ========================================================================= #


class BetaVae(Vae):

    @dataclass
    class cfg(Vae.cfg):
        # BETA SCALING:
        # =============
        # when using different loss reduction modes we need to scale beta to
        # preserve the ratio between loss components, by scaling beta.
        #   -- for loss_reduction='mean' we usually have:
        #      loss = mean_recon_loss + beta * mean_kl_loss
        #   -- for loss_reduction='mean_sum' we usually have:
        #      loss = (H*W*C) * mean_recon_loss + beta * (z_size) * mean_kl_loss
        # So when switching from one mode to the other, we need to scale beta to preserve these loss ratios.
        #   -- 'mean_sum' to 'mean':
        #      beta <- beta * (z_size) / (H*W*C)
        #   -- 'mean' to 'mean_sum':
        #      beta <- beta * (H*W*C) / (z_size)
        # We obtain an equivalent beta for 'mean_sum' to 'mean':
        #   -- given values: beta=4 for 'mean_sum', with (H*W*C)=(64*64*3) and z_size=9
        #      beta = beta * ((z_size) / (H*W*C))
        #          ~= 4 * 0.0007324
        #          ~= 0,003
        beta: float = 0.003  # approximately equal to mean_sum beta of 4

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        assert self.cfg.beta >= 0, 'beta must be >= 0'

    def training_regularize_kl(self, kl_loss):
        return self.cfg.beta * kl_loss


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
