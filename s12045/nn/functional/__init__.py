#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) CVPR-2022 Submission 12045 Authors
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

from s12045.nn.functional._conv2d import torch_conv2d_channel_wise
from s12045.nn.functional._conv2d import torch_conv2d_channel_wise_fft

from s12045.nn.functional._conv2d_kernels import get_kernel_size
from s12045.nn.functional._conv2d_kernels import torch_gaussian_kernel
from s12045.nn.functional._conv2d_kernels import torch_gaussian_kernel_2d
from s12045.nn.functional._conv2d_kernels import torch_box_kernel
from s12045.nn.functional._conv2d_kernels import torch_box_kernel_2d

from s12045.nn.functional._correlation import torch_cov_matrix
from s12045.nn.functional._correlation import torch_corr_matrix
from s12045.nn.functional._correlation import torch_rank_corr_matrix
from s12045.nn.functional._correlation import torch_pearsons_corr_matrix
from s12045.nn.functional._correlation import torch_spearmans_corr_matrix

from s12045.nn.functional._dct import torch_dct
from s12045.nn.functional._dct import torch_idct
from s12045.nn.functional._dct import torch_dct2
from s12045.nn.functional._dct import torch_idct2

from s12045.nn.functional._mean import torch_mean_generalized
from s12045.nn.functional._mean import torch_mean_quadratic
from s12045.nn.functional._mean import torch_mean_geometric
from s12045.nn.functional._mean import torch_mean_harmonic

from s12045.nn.functional._other import torch_normalize
from s12045.nn.functional._other import torch_nan_to_num
from s12045.nn.functional._other import torch_unsqueeze_l
from s12045.nn.functional._other import torch_unsqueeze_r

from s12045.nn.functional._pca import torch_pca_eig
from s12045.nn.functional._pca import torch_pca_svd
from s12045.nn.functional._pca import torch_pca

# from s12045.nn.functional._util_generic import TypeGenericTensor
# from s12045.nn.functional._util_generic import TypeGenericTorch
# from s12045.nn.functional._util_generic import TypeGenericNumpy
# from s12045.nn.functional._util_generic import generic_as_int32
# from s12045.nn.functional._util_generic import generic_max
# from s12045.nn.functional._util_generic import generic_min
# from s12045.nn.functional._util_generic import generic_shape
# from s12045.nn.functional._util_generic import generic_ndim
