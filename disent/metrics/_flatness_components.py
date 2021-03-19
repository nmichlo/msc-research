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


import logging

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from disent.dataset.groundtruth import GroundTruthDataset
from disent.metrics._flatness import encode_all_along_factor
from disent.metrics._flatness import encode_all_factors
from disent.metrics._flatness import get_device
from disent.util import iter_chunks
from disent.util import to_numpy
from disent.util.math import torch_corr_matrix
from disent.util.math import torch_mean_generalized
from disent.util.math import torch_nan_to_num
from disent.util.math import torch_rank_corr_matrix
from disent.util.math import torch_tril_mean


log = logging.getLogger(__name__)


# ========================================================================= #
# flatness                                                                  #
# ========================================================================= #


def metric_flatness_components(
        ground_truth_dataset: GroundTruthDataset,
        representation_function: callable,
        factor_repeats: int = 1024,
        batch_size: int = 64,
):
    """
    Computes the dual flatness metrics (ordering & linearity):
        swap_ratio: percent of correctly ordered ground truth factors in the latent space
        ave_corr: average of the correlation matrix (Pearson's) for latent traversals
        ave_rank_corr: average of the rank correlation matrix (Spearman's) for latent traversals

    Args:
      ground_truth_dataset: GroundTruthData to be sampled from.
      representation_function: Function that takes observations as input and outputs a dim_representation sized representation for each observation.
      factor_repeats: how many times to repeat a traversal along each factors, these are then averaged together.
      batch_size: Batch size to process at any time while generating representations, should not effect metric results.
    Returns:
      Dictionary with metrics
    """
    fs_measures = aggregate_measure_distances_along_all_factors(ground_truth_dataset, representation_function, repeats=factor_repeats, batch_size=batch_size)
    # get the means
    def multi_mean(name, ps=('arithmetic', 'geometric', 'harmonic')):
        return {
            f'dual_flatness.{name}.{p}': to_numpy(torch_mean_generalized(fs_measures[name].to(torch.float64), dim=0, p=p).to(torch.float32))
            for p in ps
        }
    results = {
        **multi_mean('ave_corr',          ps=('arithmetic', 'geometric')),
        **multi_mean('ave_rank_corr',     ps=('arithmetic', 'geometric')),
        # traversals
        # **multi_mean('swap_ratio_l1',     ps=('arithmetic', 'geometric')),
        **multi_mean('swap_ratio_l2',     ps=('arithmetic', 'geometric')),
        # any pairs
        # **multi_mean('ran_swap_ratio_l1', ps=('arithmetic',)),
        **multi_mean('ran_swap_ratio_l2', ps=('arithmetic',)),
    }
    # convert values from torch
    return {k: float(v) for k, v in results.items()}


def aggregate_measure_distances_along_all_factors(
        ground_truth_dataset,
        representation_function,
        repeats: int,
        batch_size: int,
) -> dict:
    # COMPUTE AGGREGATES FOR EACH FACTOR
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    fs_measures = default_collate([
        aggregate_measure_distances_along_factor(ground_truth_dataset, representation_function, f_idx=f_idx, repeats=repeats, batch_size=batch_size)
        for f_idx in range(ground_truth_dataset.num_factors)
    ])

    # COMPUTE RANDOM
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    values = []
    num_samples = int(np.mean(ground_truth_dataset.factor_sizes) * repeats)
    for idxs in iter_chunks(range(num_samples), batch_size):
        # encode factors
        factors = ground_truth_dataset.sample_factors(size=len(idxs))
        zs = encode_all_factors(ground_truth_dataset, representation_function, factors, batch_size=batch_size)
        # get random triplets from factors
        rai, rpi, rni = np.random.randint(0, len(factors), size=(3, len(factors) * 4))
        rai, rpi, rni = reorder_by_factor_dist(factors, rai, rpi, rni)
        # check differences
        swap_ratio_l1, swap_ratio_l2 = compute_swap_ratios(zs[rai], zs[rpi], zs[rni])
        values.append({'ran_swap_ratio_l1': swap_ratio_l1, 'ran_swap_ratio_l2': swap_ratio_l2})
    # return all
    return {
        **fs_measures,
        **default_collate(values),
    }


def reorder_by_factor_dist(factors, rai, rpi, rni):
    a_fs, p_fs, n_fs = factors[rai], factors[rpi], factors[rni]
    # sort all
    d_ap = np.linalg.norm(a_fs - p_fs, ord=1, axis=-1)
    d_an = np.linalg.norm(a_fs - n_fs, ord=1, axis=-1)
    # swap
    swap_mask = d_ap <= d_an
    rpi_NEW = np.where(swap_mask, rpi, rni)
    rni_NEW = np.where(swap_mask, rni, rpi)
    # return new
    return rai, rpi_NEW, rni_NEW


def compute_swap_ratios(a_zs, p_zs, n_zs):
    ap_delta_l1, an_delta_l1 = torch.norm(a_zs - p_zs, dim=-1, p=1), torch.norm(a_zs - n_zs, dim=-1, p=1)
    ap_delta_l2, an_delta_l2 = torch.norm(a_zs - p_zs, dim=-1, p=2), torch.norm(a_zs - n_zs, dim=-1, p=2)
    swap_ratio_l1 = (ap_delta_l1 <= an_delta_l1).to(torch.float32).mean()
    swap_ratio_l2 = (ap_delta_l2 <= an_delta_l2).to(torch.float32).mean()
    return swap_ratio_l1, swap_ratio_l2


def aggregate_measure_distances_along_factor(
        ground_truth_dataset,
        representation_function,
        f_idx: int,
        repeats: int,
        batch_size: int,
) -> dict:
    # FEED FORWARD, COMPUTE ALL
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    measures = []
    for i in range(repeats):
        # generate repeated factors, varying one factor over the entire range
        zs_traversal = encode_all_along_factor(ground_truth_dataset, representation_function, f_idx=f_idx, batch_size=batch_size)
        # check the number of swapped elements along a factor
        swap_ratio_l1, swap_ratio_l2 = compute_swap_ratios(zs_traversal[:-2], zs_traversal[1:-1], zs_traversal[2:])
        # correlations -- replace invalid values
        corr_matrix = torch.abs(torch_nan_to_num(torch_corr_matrix(zs_traversal), nan=1.0, posinf=1.0, neginf=-1.0))
        rank_corr_matrix = torch.abs(torch_nan_to_num(torch_rank_corr_matrix(zs_traversal), nan=1.0, posinf=1.0, neginf=-1.0))
        # save variables
        measures.append({
            'swap_ratio_l1': swap_ratio_l1,
            'swap_ratio_l2': swap_ratio_l2,
            'ave_corr': torch_tril_mean(corr_matrix),
            'ave_rank_corr': torch_tril_mean(rank_corr_matrix),
        })

    # AGGREGATE DATA - For each distance measure
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    measures = default_collate(measures)
    return {
        'swap_ratio_l1': measures['swap_ratio_l1'].mean(dim=0),  # shape: (repeats,) -> ()
        'swap_ratio_l2': measures['swap_ratio_l2'].mean(dim=0),  # shape: (repeats,) -> ()
        'ave_corr':      measures['ave_corr'].mean(dim=0),       # shape: (repeats,) -> ()
        'ave_rank_corr': measures['ave_rank_corr'].mean(dim=0),  # shape: (repeats,) -> ()
    }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    import pytorch_lightning as pl
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from disent.data.groundtruth import XYObjectData, XYSquaresData
    from disent.dataset.groundtruth import GroundTruthDataset, GroundTruthDatasetPairs
    from disent.frameworks.vae.unsupervised import BetaVae
    from disent.frameworks.vae.weaklysupervised import AdaVae
    from disent.frameworks.vae.supervised import TripletVae
    from disent.model.ae import EncoderConv64, DecoderConv64, AutoEncoder
    from disent.transform import ToStandardisedTensor
    from disent.util import colors
    from disent.util import Timer

    def get_str(r):
        return ', '.join(f'{k}={v:6.4f}' for k, v in r.items())

    def print_r(name, steps, result, clr=colors.lYLW, t: Timer = None):
        print(f'{clr}{name:<13} ({steps:>04}){f" {colors.GRY}[{t.pretty}]{clr}" if t else ""}: {get_str(result)}{colors.RST}')

    def calculate(name, steps, dataset, get_repr):
        global aggregate_measure_distances_along_factor
        with Timer() as t:
            r = metric_flatness_components(dataset, get_repr, factor_repeats=64, batch_size=64)
        results.append((name, steps, r))
        print_r(name, steps, r, colors.lRED, t=t)
        print(colors.GRY, '='*100, colors.RST, sep='')
        return r

    class XYOverlapData(XYSquaresData):
        def __init__(self, square_size=8, grid_size=64, grid_spacing=None, num_squares=3, rgb=True):
            if grid_spacing is None:
                grid_spacing = (square_size+1) // 2
            super().__init__(square_size=square_size, grid_size=grid_size, grid_spacing=grid_spacing, num_squares=num_squares, rgb=rgb)

    # datasets = [XYObjectData(rgb=False, palette='white'), XYSquaresData(), XYOverlapData(), XYObjectData()]
    datasets = [XYObjectData()]

    # TODO: fix for dead dimensions
    # datasets = [XYObjectData(rgb=False, palette='white')]

    results = []
    for data in datasets:

        dataset = GroundTruthDatasetPairs(data, transform=ToStandardisedTensor())
        dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, pin_memory=True)
        module = AdaVae(
            make_optimizer_fn=lambda params: Adam(params, lr=5e-4),
            make_model_fn=lambda: AutoEncoder(
                encoder=EncoderConv64(x_shape=data.x_shape, z_size=6, z_multiplier=2),
                decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
            ),
            cfg=AdaVae.cfg(beta=0.001, loss_reduction='mean')
        )

        # dataset = GroundTruthDatasetTriples(data, transform=ToStandardisedTensor(), swap_metric='manhattan')
        # dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, pin_memory=True)
        # module = TripletVae(
        #     make_optimizer_fn=lambda params: Adam(params, lr=5e-4),
        #     make_model_fn=lambda: AutoEncoder(
        #         encoder=EncoderConv64(x_shape=data.x_shape, z_size=6, z_multiplier=2),
        #         decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
        #     ),
        #     cfg=TripletVae.cfg(beta=0.001, loss_reduction='mean', triplet_p=2, triplet_scale=100)
        # )

        # we cannot guarantee which device the representation is on
        get_repr = lambda x: module.encode(x.to(module.device))
        # PHASE 1, UNTRAINED
        pl.Trainer(logger=False, checkpoint_callback=False, fast_dev_run=True, gpus=1, weights_summary=None).fit(module, dataloader)
        module = module.to('cuda')
        calculate(data.__class__.__name__, 0, dataset, get_repr)
        # PHASE 2, LITTLE TRAINING
        pl.Trainer(logger=False, checkpoint_callback=False, max_steps=256, gpus=1, weights_summary=None).fit(module, dataloader)
        calculate(data.__class__.__name__, 256, dataset, get_repr)
        # PHASE 3, MORE TRAINING
        pl.Trainer(logger=False, checkpoint_callback=False, max_steps=2048, gpus=1, weights_summary=None).fit(module, dataloader)
        calculate(data.__class__.__name__, 256+2048, dataset, get_repr)
        results.append(None)

    for result in results:
        if result is None:
            print()
            continue
        (name, steps, result) = result
        print_r(name, steps, result, colors.lYLW)