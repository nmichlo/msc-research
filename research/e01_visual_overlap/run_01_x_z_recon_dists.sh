#!/bin/bash

#
# ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
# MIT License
#
# Copyright (c) CVPR-2022 Submission 12045 Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="author_12045"
export PROJECT="final-01__gt-vs-learnt-dists"
export PARTITION="stampede"
export PARALLELISM=28

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-s12045" # 24 hours


# 1 * (3 * 6 * 4 * 2) = 144
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep' \
    \
    model=linear,vae_fc,vae_conv64 \
    \
    run_length=medium \
    metrics=all \
    \
    dataset=xyobject,xyobject_shaded,shapes3d,dsprites,cars3d,smallnorb \
    sampling=default__bb \
    framework=ae,X--adaae_os,betavae,adavae_os \
    \
    settings.framework.beta=0.0316 \
    settings.optimizer.lr=3e-4 \
    settings.model.z_size=9,25
