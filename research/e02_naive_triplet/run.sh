#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export PROJECT="exp-naive-triplet"
export PARTITION="batch"
export PARALLELISM=24

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

# 2 * (3*2*3*2=36) = 72
submit_sweep \
    framework=tvae \
    dataset=xysquares \
    \
    +DUMMY.repeat=1,2 \
    \
    system.framework.cfg_cls.triplet_margin_max=1.0,10.0 \
    system.framework.cfg_cls.triplet_scale=0.1,1.0,0.01 \
    sampling=gt_dist_factors,gt_dist_manhat,gt_dist_combined \
    system.framework.cfg_cls.triplet_p=1,2

# 2 * (3=3) = 6
submit_sweep \
    framework=tvae \
    dataset=xysquares \
    framework.name='tri-betavae' \
    \
    +DUMMY.repeat=1,2 \
    \
    sampling=gt_dist_factors,gt_dist_manhat,gt_dist_combined \
    system.framework.cfg_cls.triplet_scale=0.0

# 2 * (2*3=6) = 12
submit_sweep \
    framework=betavae,adavae \
    dataset=xysquares \
    \
    +DUMMY.repeat=1,2 \
    \
    sampling=gt_dist_factors,gt_dist_manhat,gt_dist_combined
