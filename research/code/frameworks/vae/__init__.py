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

# supervised frameworks
from research.code.frameworks.vae._supervised__adaave_tvae import AdaAveTripletVae
from research.code.frameworks.vae._supervised__adaneg_tvae import AdaNegTripletVae
from research.code.frameworks.vae._supervised__adatvae import AdaTripletVae
from research.code.frameworks.vae._supervised__badavae import BoundedAdaVae
from research.code.frameworks.vae._supervised__gadavae import GuidedAdaVae
from research.code.frameworks.vae._supervised__tbadavae import TripletBoundedAdaVae
from research.code.frameworks.vae._supervised__tgadavae import TripletGuidedAdaVae

# unsupervised frameworks
from research.code.frameworks.vae._unsupervised__dorvae import DataOverlapRankVae
from research.code.frameworks.vae._unsupervised__dotvae import DataOverlapTripletVae

# weakly supervised frameworks
from research.code.frameworks.vae._weaklysupervised__augpostriplet import AugPosTripletVae
from research.code.frameworks.vae._weaklysupervised__st_adavae import SwappedTargetAdaVae
from research.code.frameworks.vae._weaklysupervised__st_betavae import SwappedTargetBetaVae
