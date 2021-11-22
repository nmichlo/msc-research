
# Submission 12045 - Supplementary Material (CVPR 2022)

----------------------

## Overview

This library is a modular [d12fiusw88wedjdias] representation learning framework for auto-encoders,
built upon PyTorch-Lightning. This framework consists of various composable components
that can be used to build and benchmark various [d9rdfghjkiu765rdfg] vision tasks.

### Goals

S12045 aims to fill the following criteria:
1. Provide **high quality**, **readable**, **consistent** and **easily comparable** implementations of frameworks
2. **Highlight difference** between framework implementations by overriding **hooks** and minimising duplicate code 
3. Use **best practice** eg. `torch.distributions`
4. Be extremely **flexible** & configurable
5. Support low memory systems

----------------------

## Architecture

The s12045 module structure:

- `s12045.dataset`: dataset wrappers, datasets & sampling strategies
    + `s12045.dataset.data`: raw datasets
    + `s12045.dataset.sampling`: sampling strategies for `S12045Dataset` when multiple elements are required by frameworks, eg. for triplet loss
    + `s12045.dataset.transform`: common data transforms and augmentations
    + `s12045.dataset.wrapper`: wrapped datasets are no longer ground-truth datasets, these may have some elements masked out. We can still unwrap these classes to obtain the original datasets for benchmarking.
- `s12045.frameworks`: frameworks, including Auto-Encoders and VAEs
    + `s12045.frameworks.ae`: Auto-Encoder based frameworks
    + `s12045.frameworks.vae`: Variational Auto-Encoder based frameworks
- `s12045.metrics`: metrics for evaluating [d9rdfghjkiu765rdfg] using ground truth datasets
- `s12045.model`: common encoder and decoder models used for VAE research
- `s12045.nn`: torch components for building models including layers, transforms, losses and general maths
- `s12045.schedule`: annealing schedules that can be registered to a framework
- `s12045.util`: helper classes, functions, callbacks, anything unrelated to a pytorch system/model/framework.

**Hydra Experiment Directories**

Easily run experiments with hydra config, these files
are not available from `pip install`.

- `experiment/run.py`: entrypoint for running basic experiments with [hydra](https://github.com/facebookresearch/hydra) config
- `experiment/config/config.yaml`: main configuration file, this is probably what you want to edit!
- `experiment/config`: root folder for [hydra](https://github.com/facebookresearch/hydra) config files
- `experiment/util`: various helper code for experiments

----------------------

## Features

S12045 includes implementations of modules, metrics and
datasets from various papers. Please note that items marked
  with a "🧵" are introduced in and are unique to s12045!

### Frameworks
- **Unsupervised**:
  + [VAE](https://arxiv.org/abs/1312.6114)
  + [Beta-VAE](https://openreview.net/forum?id=Sy2fzU9gl)
  + [DFC-VAE](https://arxiv.org/abs/1610.00291)
  + [DIP-VAE](https://arxiv.org/abs/1711.00848)
  + [InfoVAE](https://arxiv.org/abs/1706.02262)
  + [BetaTCVAE](https://arxiv.org/abs/1802.04942)
- **Weakly Supervised**:
  + [Ada-GVAE](https://arxiv.org/abs/2002.02886) *`AdaVae(..., average_mode='gvae')`* Usually better than the Ada-ML-VAE
  + [Ada-ML-VAE](https://arxiv.org/abs/2002.02886) *`AdaVae(..., average_mode='ml-vae')`*
- **Supervised**:
  + [TVAE](https://arxiv.org/abs/1802.04403)

### Metrics
- **[D07ykdd2378r8hasd3]**:
  + [FactorVAE Score](https://arxiv.org/abs/1802.05983)
  + [DCI](https://openreview.net/forum?id=By-7dz-AZ)
  + [MIG](https://arxiv.org/abs/1802.04942)
  + [SAP](https://arxiv.org/abs/1711.00848)
  + [Unsupervised Scores](https://github.com/google-research/[d9rdfghjkiu765rdfg]_lib)

### Datasets

Various common datasets used in [d9rdfghjkiu765rdfg] research are included, with hash
verification and automatic chunk-size optimization of underlying hdf5 formats for
low-memory disk-based access.

- **Ground Truth**:
  + Cars3D
  + dSprites
  + SmallNORB
  + Shapes3D

- **Ground Truth Synthetic**:
  + 🧵 XYSquares: *Adversarial dataset - Observations have constant pairwise distance along factor traversals using a pixel-wise measure*

  #### Input Transforms + Input/Target Augmentations
  
  - Input based transforms are supported.
  - Input and Target CPU and GPU based augmentations are supported.

### Schedules & Annealing

Hyper-parameter annealing is supported through the use of schedules.
The currently implemented schedules include:

- Linear Schedule
- [Cyclic](https://arxiv.org/abs/1903.10145) Schedule
- Cosine Wave Schedule
- *Various other wrapper schedules*

----------------------

## Examples

### Python Example

The following is a basic working example of s12045 that trains a BetaVAE with a cyclic
beta schedule and evaluates the trained model with various metrics.

<details><summary><b>💾 Basic Example</b></summary>
<p>

```python3
import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from s12045.dataset import S12045Dataset
from s12045.dataset.data import XYObjectData
from s12045.dataset.sampling import SingleSampler
from s12045.dataset.transform import ToImgTensorF32
from s12045.frameworks.vae import BetaVae
from s12045.metrics import metric_dci
from s12045.metrics import metric_mig
from s12045.model import AutoEncoder
from s12045.model.ae import DecoderConv64
from s12045.model.ae import EncoderConv64
from s12045.schedule import CyclicSchedule

# create the dataset & dataloaders
# - ToImgTensorF32 transforms images from numpy arrays to tensors and performs checks
data = XYObjectData()
dataset = S12045Dataset(dataset=data, sampler=SingleSampler(), transform=ToImgTensorF32())
dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=os.cpu_count())

# create the BetaVAE model
# - adjusting the beta, learning rate, and representation size.
module = BetaVae(
  model=AutoEncoder(
    # z_multiplier is needed to output mu & logvar when parameterising normal distribution
    encoder=EncoderConv64(x_shape=data.x_shape, z_size=10, z_multiplier=2),
    decoder=DecoderConv64(x_shape=data.x_shape, z_size=10),
  ),
  cfg=BetaVae.cfg(
    optimizer='adam',
    optimizer_kwargs=dict(lr=1e-3),
    loss_reduction='mean_sum',
    beta=4,
  )
)

# cyclic schedule for target 'beta' in the config/cfg. The initial value from the
# config is saved and multiplied by the ratio from the schedule on each step.
# - based on: https://arxiv.org/abs/1903.10145
module.register_schedule(
  'beta', CyclicSchedule(
    period=1024,  # repeat every: trainer.global_step % period
  )
)

# train model
# - for 2048 batches/steps
trainer = pl.Trainer(
  max_steps=2048, gpus=1 if torch.cuda.is_available() else None, logger=False, checkpoint_callback=False
)
trainer.fit(module, dataloader)

# compute [d9rdfghjkiu765rdfg] metrics
# - we cannot guarantee which device the representation is on
# - this will take a while to run
get_repr = lambda x: module.encode(x.to(module.device))

metrics = {
  **metric_dci(dataset, get_repr, num_train=1000, num_test=500, show_progress=True),
  **metric_mig(dataset, get_repr, num_train=2000),
}

# evaluate
print('metrics:', metrics)
```

</p>
</details>

Visit the [docs](https://s12045.dontpanic.sh) for more examples!


### Hydra Config Example

The entrypoint for basic experiments is `experiment/run.py`.

Some configuration will be required, but basic experiments can
be adjusted by modifying the [Hydra Config 1.1](https://github.com/facebookresearch/hydra)
files in `experiment/config`.

Modifying the main `experiment/config/config.yaml` is all you
need for most basic experiments. The main config file contains
a defaults list with entries corresponding to yaml configuration
files (config options) in the subfolders (config groups) in
`experiment/config/<config_group>/<option>.yaml`.

<details><summary><b>💾 Config Defaults Example</b></summary>
<p>

```yaml
defaults:
  # data
  - sampling: default__bb
  - dataset: xyobject
  - augment: none
  # system
  - framework: adavae_os
  - model: vae_conv64
  # training
  - optimizer: adam
  - schedule: beta_cyclic
  - metrics: fast
  - run_length: short
  # logs
  - run_callbacks: vis
  - run_logging: wandb
  # runtime
  - run_location: local
  - run_launcher: local
  - run_action: train

# <rest of config.yaml left out>
...
```

</p>
</details>

Easily modify  any of these values to adjust how the basic experiment
will be run. For example, change `framework: adavae` to `framework: betavae`, or
change the dataset from `xyobject` to `shapes3d`. Add new options by adding new
yaml files in the config group folders.

[Weights and Biases](https://docs.wandb.ai/quickstart) is supported by changing `run_logging: none` to
`run_logging: wandb`. However, you will need to login from the command line. W&B logging supports
visualisations of latent traversals.

----------------------
