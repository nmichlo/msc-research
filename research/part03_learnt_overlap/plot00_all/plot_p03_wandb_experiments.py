#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2022 Nathan Juraj Michlo
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
import os
from typing import Optional
from typing import Sequence

import pandas as pd
import seaborn as sns
from cachier import cachier as _cachier
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

import research.code.util as H
from disent.util.function import wrapped_partial
from disent.util.profiling import Timer
from research.code.util._wandb_plots import drop_non_unique_cols
from research.code.util._wandb_plots import drop_unhashable_cols
from research.code.util._wandb_plots import load_runs as _load_runs


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


DF = pd.DataFrame

# cachier instance
CACHIER: _cachier = wrapped_partial(_cachier, cache_dir=os.path.join(os.path.dirname(__file__), 'plots/.cache'))


@CACHIER()
def load_runs(project: str, include_history: bool = False):
    return _load_runs(project=project, include_history=include_history)


def clear_cache(clear_data=True, clear_wandb=False):
    from research.code.util._wandb_plots import clear_runs_cache
    if clear_wandb:
        clear_runs_cache()
    if clear_data:
        load_runs.clear_cache()


# ========================================================================= #
# Prepare Data                                                              #
# ========================================================================= #


# common keys
K_GROUP     = 'Run Group'
K_DATASET   = 'Dataset'
K_FRAMEWORK = 'Framework'
K_BETA      = 'Beta'
K_LOSS      = 'Recon. Loss'
K_Z_SIZE    = 'Latent Dims.'
K_REPEAT    = 'Repeat'
K_STATE     = 'State'
K_LR        = 'Learning Rate'
K_SCHEDULE  = 'Schedule'
K_SAMPLER   = 'Sampler'
K_ADA_MODE  = 'Threshold Mode'

K_MIG_END = 'MIG Score\n(End)'
K_DCI_END = 'DCI Score\n(End)'
K_MIG_MAX = 'MIG Score'
K_DCI_MAX = 'DCI Score'
K_LCORR_GT_F   = 'Linear Corr.\n(factors)'
K_RCORR_GT_F   = 'Rank Corr.\n(factors)'
K_LCORR_GT_G   = 'Global Linear Corr.\n(factors)'
K_RCORR_GT_G   = 'Global Rank Corr.\n(factors)'
K_LCORR_DATA_F = 'Linear Corr.\n(data)'
K_RCORR_DATA_F = 'Rank Corr.\n(data)'
K_LCORR_DATA_G = 'Global Linear Corr.\n(data)'
K_RCORR_DATA_G = 'Global Rank Corr.\n(data)'
K_AXIS      = 'Axis Ratio'
K_LINE      = 'Linear Ratio'

K_TRIPLET_SCALE  = 'Triplet Scale'   # framework.cfg.triplet_margin_max
K_TRIPLET_MARGIN = 'Triplet Margin'  # framework.cfg.triplet_scale
K_TRIPLET_P      = 'Triplet P'       # framework.cfg.triplet_p
K_DETACH         = 'Detached'        # framework.cfg.detach_decoder
K_TRIPLET_MODE   = 'Triplet Mode'    # framework.cfg.triplet_loss


def load_general_data(
    project: str,
    include_history: bool = False,
    keep_cols: Sequence[str] = None,
    drop_unhashable: bool = False,
    drop_non_unique: bool = False,
):
    # keep columns
    if keep_cols is None:
        keep_cols = []
    keep_cols = list(keep_cols)
    if include_history:
        keep_cols = ['history'] + keep_cols
    # load data
    df = load_runs(project, include_history=include_history)
    # process data
    with Timer('processing data'):
        # rename columns
        df = df.rename(columns={
            'settings/optimizer/lr':                K_LR,
            'EXTRA/tags':                           K_GROUP,
            'dataset/name':                         K_DATASET,
            'framework/name':                       K_FRAMEWORK,
            'settings/framework/beta':              K_BETA,
            'settings/framework/recon_loss':        K_LOSS,
            'settings/model/z_size':                K_Z_SIZE,
            'DUMMY/repeat':                         K_REPEAT,
            'state':                                K_STATE,
            'final_metric/mig.discrete_score.max':  K_MIG_END,
            'final_metric/dci.disentanglement.max': K_DCI_END,
            'epoch_metric/mig.discrete_score.max':  K_MIG_MAX,
            'epoch_metric/dci.disentanglement.max': K_DCI_MAX,
            # scores
            'epoch_metric/distances.lcorr_ground_latent.l1.factor.max': K_LCORR_GT_F,
            'epoch_metric/distances.rcorr_ground_latent.l1.factor.max': K_RCORR_GT_F,
            'epoch_metric/distances.lcorr_ground_latent.l1.global.max': K_LCORR_GT_G,
            'epoch_metric/distances.rcorr_ground_latent.l1.global.max': K_RCORR_GT_G,
            'epoch_metric/distances.lcorr_latent_data.l2.factor.max':   K_LCORR_DATA_F,
            'epoch_metric/distances.rcorr_latent_data.l2.factor.max':   K_RCORR_DATA_F,
            'epoch_metric/distances.lcorr_latent_data.l2.global.max':   K_LCORR_DATA_G,
            'epoch_metric/distances.rcorr_latent_data.l2.global.max':   K_RCORR_DATA_G,
            'epoch_metric/linearity.axis_ratio.var.max':                K_AXIS,
            'epoch_metric/linearity.linear_ratio.var.max':              K_LINE,
            # adaptive methods
            'schedule/name': K_SCHEDULE,
            'sampling/name': K_SAMPLER,
            'framework/cfg/ada_thresh_mode': K_ADA_MODE,
            # triplet experiments
            'framework/cfg/triplet_margin_max': K_TRIPLET_MARGIN,
            'framework/cfg/triplet_scale':      K_TRIPLET_SCALE,
            'framework/cfg/triplet_p':          K_TRIPLET_P,
            'framework/cfg/detach_decoder':     K_DETACH,
            'framework/cfg/triplet_loss':       K_TRIPLET_MODE,
        })
        # filter out unneeded columns
        if drop_unhashable:
            df, dropped_hash = drop_unhashable_cols(df, skip=keep_cols)
        if drop_non_unique:
            df, dropped_diverse = drop_non_unique_cols(df, skip=keep_cols)
    return df


def rename_entries(df: pd.DataFrame):
    df = df.copy()
    # replace values in the df
    for key, value, new_value in [
        (K_DATASET,  'xysquares_minimal',        'xysquares'),
        (K_SCHEDULE, 'adanegtvae_up_all_full',   'Schedule: Both (strong)'),
        (K_SCHEDULE, 'adanegtvae_up_all',        'Schedule: Both (weak)'),
        (K_SCHEDULE, 'adanegtvae_up_ratio_full', 'Schedule: Weight (strong)'),
        (K_SCHEDULE, 'adanegtvae_up_ratio',      'Schedule: Weight (weak)'),
        (K_SCHEDULE, 'adanegtvae_up_thresh',     'Schedule: Threshold'),
        (K_TRIPLET_MODE, 'triplet', 'Triplet Loss (Hard Margin)'),
        (K_TRIPLET_MODE, 'triplet_soft', 'Triplet Loss (Soft Margin)'),
        (K_TRIPLET_P, 1, 'L1 Distance'),
        (K_TRIPLET_P, 2, 'L2 Distance'),
        # (K_SAMPLER, 'gt_dist__manhat',        'Ground-Truth Dist Sampling'),
        # (K_SAMPLER, 'gt_dist__manhat_scaled', 'Ground-Truth Dist Sampling (Scaled)'),
    ]:
        if key in df.columns:
            df[key].replace(value, new_value, inplace=True)
            # df.loc[df[key] == value, key] = new_value
    return df


# ========================================================================= #
# Plot Experiments                                                          #
# ========================================================================= #


PINK = '#FE375F'     # usually: Beta-VAE
PURPLE = '#5E5BE5'   # maybe:   Ada-TVAE
LPURPLE = '#b0b6ff'  # maybe:   Ada-TVAE (alt)
BLUE = '#1A93FE'     # maybe:   TVAE
LBLUE = '#63D2FE'
ORANGE = '#FE9F0A'   # usually: Ada-VAE
GREEN = '#2FD157'

LGREEN = '#9FD911'   # usually: MSE
LBLUE2 = '#36CFC8'   # usually: MSE-Overlap


# ========================================================================= #
# Experiment 3                                                              #
# ========================================================================= #


def plot_e03_different_gt_representations(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    color_entangled_data: str = LGREEN,
    color_disentangled_data: str = LBLUE2,
    metrics: Sequence[str] = (K_MIG_MAX, K_DCI_MAX, K_RCORR_GT_F, K_RCORR_DATA_F, K_AXIS, K_LINE),
):
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    df = load_general_data(f'{os.environ["WANDB_USER"]}/MSC-p03e03_different-gt-representations', keep_cols=(K_GROUP, K_Z_SIZE))
    # select run groups
    #     +DUMMY.repeat=1,2,3 \
    #     settings.framework.beta=0.001,0.00316,0.01,0.0316 \
    #     framework=betavae,adavae_os \
    #     settings.model.z_size=9 \
    #     dataset=xyobject,xyobject_shaded \
    df = df[df[K_GROUP].isin(['sweep_different-gt-repr_basic-vaes'])]
    df = df.sort_values([K_DATASET, K_FRAMEWORK, K_BETA, K_REPEAT])
    # print common key values
    print('K_GROUP:    ', list(df[K_GROUP].unique()))
    print('K_DATASET:  ', list(df[K_DATASET].unique()))
    print('K_FRAMEWORK:', list(df[K_FRAMEWORK].unique()))
    print('K_BETA:     ', list(df[K_BETA].unique()))
    print('K_Z_SIZE:   ', list(df[K_Z_SIZE].unique()))
    print('K_REPEAT:   ', list(df[K_REPEAT].unique()))
    print('K_STATE:    ', list(df[K_STATE].unique()))
    # rename more stuff
    df = rename_entries(df)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    orig = df
    # select runs
    # df = df[df[K_STATE] == 'finished']
    # [1.0, 0.316, 0.1, 0.0316, 0.01, 0.00316, 0.001, 0.000316]
    # df = df[(0.000316 < df[K_BETA]) & (df[K_BETA] < 1.0)]
    print('NUM', len(orig), '->', len(df))
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    df[K_DATASET].replace('xysquares_minimal', 'XYSquares', inplace=True)
    df[K_DATASET].replace('smallnorb', 'NORB', inplace=True)
    df[K_DATASET].replace('cars3d', 'Cars3D', inplace=True)
    df[K_DATASET].replace('3dshapes', 'Shapes3D', inplace=True)
    df[K_DATASET].replace('dsprites', 'dSprites', inplace=True)
    df[K_DATASET].replace('xyobject', 'XYObject', inplace=True)
    df[K_DATASET].replace('xyobject_shaded', 'XYObject (Shades)', inplace=True)
    df[K_FRAMEWORK].replace('adavae_os', 'Ada-GVAE', inplace=True)
    df[K_FRAMEWORK].replace('betavae', 'Beta-VAE', inplace=True)

    PALLETTE = {
        'XYObject': color_entangled_data,
        'XYObject (Shades)': color_disentangled_data,
    }

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    fig, axs = plt.subplots(2, len(metrics) // 2, figsize=(len(metrics)//2*3.75, 2*2.7))
    axs = axs.flatten()
    # PLOT
    for i, (key, ax) in enumerate(zip(metrics, axs)):
        assert key in df.columns, f'{repr(key)} not in {sorted(df.columns)}'
        sns.violinplot(data=df, ax=ax, x=K_FRAMEWORK, y=key, hue=K_DATASET, palette=PALLETTE, split=True, cut=0, width=0.75, scale='width', inner='quartile')
        ax.set_ylim([-0.1, 1.1])
        ax.set_ylim([0, None])
        if i == len(axs)-1:
            ax.legend(bbox_to_anchor=(0.05, 0.175), fontsize=12, loc='lower left', labelspacing=0.1)
            ax.set_xlabel(None)
            # ax.set_ylabel('Minimum Recon. Loss')
        else:
            if ax.get_legend():
                ax.get_legend().remove()
            ax.set_xlabel(None)
            # ax.set_ylabel(None)
    # PLOT:
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show, dpi=300)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    return fig, axs

# ========================================================================= #
# Experiment 3                                                              #
# ========================================================================= #


def plot_e04_random_external_factors(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    mode: str = 'fg',
    # color_entangled_data: str = LGREEN,
    # color_disentangled_data: str = LBLUE2,
    reg_order: int = 1,
    color_betavae: str = PINK,
    color_adavae: str = ORANGE,
    metrics: Sequence[str] = (K_MIG_MAX, K_DCI_MAX, K_RCORR_GT_F, K_RCORR_DATA_F, K_AXIS, K_LINE),
):
    K_IM_MODE = 'dataset/data/mode'
    K_IM_VIS = 'Visibility %'

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    df = load_general_data(f'{os.environ["WANDB_USER"]}/MSC-p03e04_random-external-factors', keep_cols=(K_GROUP, K_Z_SIZE))
    df = df.rename(columns={'dataset/data/visibility': K_IM_VIS})
    # filter the groups
    # -- FIX_ADA_RSYNC: adavae_os
    # -- FIX:           betavae
    df = df[df[K_GROUP].isin(['sweep_imagenet_dsprites_FIX_ADA_RSYNC', 'sweep_imagenet_dsprites_FIX'])]
    # select run groups
    df = df.sort_values([K_DATASET, K_FRAMEWORK, K_IM_MODE, K_IM_VIS])
    df = df[df[K_FRAMEWORK].isin(['adavae_os', 'betavae'])]
    df = df[df[K_DATASET].isin(['dsprites', f'dsprites_imagenet_{mode}_25', f'dsprites_imagenet_{mode}_50', f'dsprites_imagenet_{mode}_75', f'dsprites_imagenet_{mode}_100'])]
    # rename more stuff
    df = rename_entries(df)
    # print common key values
    print('K_GROUP:    ', list(df[K_GROUP].unique()))
    print('K_DATASET:  ', list(df[K_DATASET].unique()))
    print('K_FRAMEWORK:', list(df[K_FRAMEWORK].unique()))
    print('K_BETA:     ', list(df[K_BETA].unique()))
    print('K_Z_SIZE:   ', list(df[K_Z_SIZE].unique()))
    print('K_REPEAT:   ', list(df[K_REPEAT].unique()))
    print('K_STATE:    ', list(df[K_STATE].unique()))
    print('K_IM_VIS:   ', list(df[K_IM_VIS].unique()))
    print('K_IM_MODE:  ', list(df[K_IM_MODE].unique()))
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    # df = df[df[K_STATE].isin(['finished'])]
    # df = df[df[K_STATE].isin(['finished', 'running'])]

    # replace unset values
    df.loc[df[K_DATASET] == 'dsprites', K_IM_VIS] = 0
    df.loc[df[K_DATASET] == 'dsprites', K_IM_MODE] = mode

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    orig = df
    # select runs
    # select adavae
    data_adavae = df[(df[K_FRAMEWORK] == 'adavae_os')]
    data_betavae = df[(df[K_FRAMEWORK] == 'betavae')]
    print('ADAGVAE', len(orig), '->', len(data_adavae))
    print('BETAVAE', len(orig), '->', len(data_betavae))
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    print(data_adavae[K_MIG_MAX])
    print(data_adavae[K_IM_VIS])

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    # fig, axs = plt.subplots(1, len(metrics), figsize=(len(metrics) * 2.7*1.1, 3.33*1.1), squeeze=False)
    fig, axs = plt.subplots(2, len(metrics) // 2, figsize=(len(metrics)//2 * 2.9, 2*3.3))
    axs = axs.flatten()
    # Legend entries
    marker_ada  = mlines.Line2D([], [], color=color_adavae,  marker='o', markersize=11.5, label='Ada-GVAE')
    marker_beta = mlines.Line2D([], [], color=color_betavae, marker='X', markersize=11.5, label='Beta-VAE')  # why does 'x' not work? only 'X'?
    # PLOT: MIG
    for x, metric_key in enumerate(metrics):
        ax = axs[x]
        sns.regplot(ax=ax, x=K_IM_VIS, y=metric_key, data=data_adavae,  seed=777, order=reg_order, robust=False, color=color_adavae,  marker='o')
        sns.regplot(ax=ax, x=K_IM_VIS, y=metric_key, data=data_betavae, seed=777, order=reg_order, robust=False, color=color_betavae, marker='x', line_kws=dict(linestyle='dashed'))
        if x == 0:
            ax.legend(handles=[marker_beta, marker_ada], fontsize=14)
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlim([-5, 105])
    # PLOT:
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    return fig, axs

# ========================================================================= #
# Entrypoint                                                                #
# ========================================================================= #


def plot_e01_learnt_loss_with_vaes(
    rel_path: Optional[str] = None,
    save: bool = True,
    show: bool = True,
    color_betavae: str = PINK,
    color_adavae: str = ORANGE,
    # color_mse_xy8: str = GREEN,
    # color_mse_box: str = BLUE,
    # color_mse_gau: str = PURPLE,
    # color_mse: str = PINK,
    # color_entangled_data: str = LGREEN,
    # color_disentangled_data: str = LBLUE2,
    metrics: Sequence[str] = (K_MIG_MAX, K_DCI_MAX, K_RCORR_GT_F, K_RCORR_DATA_F, K_AXIS, K_LINE),
):
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    df = load_general_data(f'{os.environ["WANDB_USER"]}/MSC-p03e02_learnt-loss-with-vaes', keep_cols=(K_GROUP, K_Z_SIZE))
    # select run groups
    #     +DUMMY.repeat=1,2,3,4,5 \
    #     framework=betavae,adavae_os \
    #     settings.framework.beta=0.0316,0.0001 \
    #     settings.framework.recon_loss='mse','mse_gau_r31_l1.0_k3969.0_norm_sum','mse_box_r31_l1.0_k3969.0_norm_sum','mse_xy8_abs63_l1.0_k1.0_norm_none' \
    df = df[df[K_GROUP].isin(['MSC_sweep_losses', 'MSC_sweep_losses_ALT', 'MSC_sweep_losses_XY8R31'])]  # 'MSC_sweep_losses', 'MSC_sweep_losses_XY8R31', 'MSC_sweep_losses_ALT'
    df = df.sort_values([K_LOSS, K_FRAMEWORK, K_BETA, K_REPEAT])
    # print common key values
    print('K_GROUP:    ', list(df[K_GROUP].unique()))
    print('K_DATASET:  ', list(df[K_DATASET].unique()))
    print('K_FRAMEWORK:', list(df[K_FRAMEWORK].unique()))
    print('K_LOSS:     ', list(df[K_LOSS].unique()))
    print('K_BETA:     ', list(df[K_BETA].unique()))
    print('K_Z_SIZE:   ', list(df[K_Z_SIZE].unique()))
    print('K_REPEAT:   ', list(df[K_REPEAT].unique()))
    print('K_STATE:    ', list(df[K_STATE].unique()))
    print('K_LOSS:     ', list(df[K_LOSS].unique()))
    # rename more stuff
    df = rename_entries(df)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    orig = df
    # select runs
    df = df[df[K_STATE].isin(['finished'])].copy()
    print('NUM', len(orig), '->', len(df))
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    df[K_DATASET].replace('xysquares_minimal', 'XYSquares', inplace=True)
    df[K_DATASET].replace('smallnorb', 'NORB', inplace=True)
    df[K_DATASET].replace('cars3d', 'Cars3D', inplace=True)
    df[K_DATASET].replace('3dshapes', 'Shapes3D', inplace=True)
    df[K_DATASET].replace('dsprites', 'dSprites', inplace=True)
    df[K_DATASET].replace('xyobject', 'XYObject', inplace=True)
    df[K_DATASET].replace('xyobject_shaded', 'XYObject (Shades)', inplace=True)
    df[K_FRAMEWORK].replace('adavae_os', 'Ada-GVAE', inplace=True)
    df[K_FRAMEWORK].replace('betavae', 'Beta-VAE', inplace=True)

    N_mse = 'none'
    N_gau = 'gau'
    N_box = 'box'
    N_xy8r31 = 'xy8'
    N_xy8r63 = 'xy8_r63'

    N_mse = 'MSE'
    N_gau = 'MSE\n(gau)'
    N_box = 'MSE\n(box)'
    N_xy8r31 = 'MSE\n(xy8)'
    N_xy8r63 = 'MSE\n(xy8,r63)'

    df[K_LOSS].replace('mse_xy8_abs31_l1.0_k1.0_norm_none', N_xy8r31, inplace=True)
    df[K_LOSS].replace('mse_xy8_abs63_l1.0_k1.0_norm_none', N_xy8r63, inplace=True)
    df[K_LOSS].replace('mse_box_r31_l1.0_k3969.0_norm_sum', N_box,  inplace=True)
    df[K_LOSS].replace('mse_gau_r31_l1.0_k3969.0_norm_sum', N_gau,  inplace=True)
    df[K_LOSS].replace('mse',                               N_mse, inplace=True)

    df['_sort_'] = list(df[K_LOSS])
    df['_sort_'].replace(N_mse, 1, inplace=True)
    df['_sort_'].replace(N_gau, 2, inplace=True)
    df['_sort_'].replace(N_box, 3, inplace=True)
    df['_sort_'].replace(N_xy8r31, 4, inplace=True)
    df['_sort_'].replace(N_xy8r63, 5, inplace=True)

    df = df[df[K_LOSS].isin([N_mse, N_gau, N_box, N_xy8r31])]

    # df = df[df[K_BETA].isin([
    #     0.0001,
    #     0.000316,
    #     0.001,
    #     0.00316,
    #     0.01,
    #     0.0316
    # ])]

    df = df.sort_values(['_sort_', K_LOSS, K_FRAMEWORK, K_BETA])

    print('NUM?', len(df))
    df = df[
        ((df[K_LOSS] == N_mse)     & ( df[K_BETA].isin([0.0001, 0.000316])))
      # ((df[K_LOSS] == N_gau)     & (~df[K_BETA].isin([0.0001, 0.000316, 0.001, 0.00316, 0.01, 0.0316])))
      | ((df[K_LOSS] == N_gau)     & (~df[K_BETA].isin([0.0001, 0.000316,                 0.01, 0.0316])))
      | ((df[K_LOSS] == N_box)     & (~df[K_BETA].isin([0.0001, 0.000316, 0.001, 0.00316])))
      | ((df[K_LOSS] == N_xy8r31)  & (~df[K_BETA].isin([0.0001, 0.000316, 0.001, 0.00316])))
    ]
    print('NUM?', len(df))

    PALLETTE = {
        # 'MSE (learnt)': color_mse_xy8,
        # 'MSE (box)': color_mse_box,
        # 'MSE (gaussian)': color_mse_gau,
        # 'MSE': color_mse,
        'Beta-VAE': color_betavae,
        'Ada-GVAE': color_adavae,
    }

    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #
    fig, axs = plt.subplots(2, len(metrics) // 2, figsize=(len(metrics) // 2 * 3.75, 2 * 3.0))
    axs = axs.flatten()
    # PLOT
    for i, (key, ax) in enumerate(zip(metrics, axs)):
        assert key in df.columns, f'{repr(key)} not in {sorted(df.columns)}'
        sns.violinplot(data=df, ax=ax, x=K_LOSS, y=key, hue=K_FRAMEWORK, palette=PALLETTE, split=True, cut=0, width=0.75, scale='width', inner='quartile')
        ax.set_ylim([-0.1, 1.1])
        ax.set_ylim([0, None])
        if i == len(axs) - 1:
            # ax.legend(bbox_to_anchor=(0.05, 0.175), fontsize=12, loc='lower left', labelspacing=0.1)
            ax.legend(fontsize=12, bbox_to_anchor=(1.0, 0.05), loc='lower right', labelspacing=0.1)
            ax.set_xlabel(None)
            # ax.set_ylabel('Minimum Recon. Loss')
        else:
            if ax.get_legend():
                ax.get_legend().remove()
            ax.set_xlabel(None)
            # ax.set_ylabel(None)
    # PLOT:
    fig.tight_layout()
    H.plt_rel_path_savefig(rel_path, save=save, show=show, dpi=300)
    # ~=~=~=~=~=~=~=~=~=~=~=~=~ #

    return fig, axs


# ========================================================================= #
# Entrypoint                                                                #
# ========================================================================= #


if __name__ == '__main__':

    assert 'WANDB_USER' in os.environ, 'specify "WANDB_USER" environment variable'

    logging.basicConfig(level=logging.INFO)

    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), '../../code/util/gadfly.mplstyle'))

    # clear_cache(clear_data=True, clear_wandb=True)
    # clear_cache(clear_data=True, clear_wandb=False)

    def main():
        plot_e01_learnt_loss_with_vaes(rel_path='plots/p03e01_learnt-loss-with-vaes', show=True)

        plot_e03_different_gt_representations(rel_path='plots/p03e03_different-gt-representations', show=True)

        plot_e04_random_external_factors(rel_path='plots/p03e04_random-external-factors__fg', mode='fg', show=True)
        plot_e04_random_external_factors(rel_path='plots/p03e04_random-external-factors__bg', mode='bg', show=True)

    main()


# ========================================================================= #
# DONE                                                                      #
# ========================================================================= #
