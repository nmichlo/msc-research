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

# custom episodes -- base
from s12045.dataset.data._episodes import BaseEpisodesData
from s12045.dataset.data._episodes__custom import EpisodesPickledData
from s12045.dataset.data._episodes__custom import EpisodesDownloadZippedPickledData

# raw -- groundtruth
from s12045.dataset.data._groundtruth import ArrayGroundTruthData
from s12045.dataset.data._groundtruth import SelfContainedHdf5GroundTruthData

# raw
from s12045.dataset.data._raw import ArrayDataset
from s12045.dataset.data._raw import Hdf5Dataset

# groundtruth -- base
from s12045.dataset.data._groundtruth import GroundTruthData
from s12045.dataset.data._groundtruth import DiskGroundTruthData
from s12045.dataset.data._groundtruth import NumpyFileGroundTruthData
from s12045.dataset.data._groundtruth import Hdf5GroundTruthData

# groundtruth -- impl
from s12045.dataset.data._groundtruth__cars3d import Cars3dData
from s12045.dataset.data._groundtruth__dsprites import DSpritesData
from s12045.dataset.data._groundtruth__dsprites_imagenet import DSpritesImagenetData  # pragma: delete-on-release
from s12045.dataset.data._groundtruth__mpi3d import Mpi3dData
from s12045.dataset.data._groundtruth__norb import SmallNorbData
from s12045.dataset.data._groundtruth__shapes3d import Shapes3dData

# groundtruth -- impl synthetic
from s12045.dataset.data._groundtruth__xyblocks import XYBlocksData           # pragma: delete-on-release
from s12045.dataset.data._groundtruth__xyobject import XYObjectData
from s12045.dataset.data._groundtruth__xyobject import XYObjectShadedData
from s12045.dataset.data._groundtruth__xysquares import XYSquaresData         # pragma: delete-on-release
from s12045.dataset.data._groundtruth__xysquares import XYSquaresMinimalData  # pragma: delete-on-release
from s12045.dataset.data._groundtruth__xcolumns import XColumnsData           # pragma: delete-on-release
