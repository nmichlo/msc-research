
# Submission 12045 - Supplementary Material (CVPR 2022)

----------------------

## Overview

**Directory Structure**

- `s12045`: disentanglement framework that we contribute from this work (please see `README-LIB.md` for more detailed usage)
- `experiment`: configuration and runner files calling the s12045 framework
- `research`: actual experimental setup, sweeps, plotting and scripts for generating the results for this CVPR submission.

## Setup

Python 3.8 was used to run experiments on Ubuntu 18

- Requirements were installed in order:
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install -r requirements-experiment.txt
  pip install -r requirements-research.txt
  pip install -r requirements-test.txt
  ```

- Check that everything is working by running pytest `PYTHONPATH=. python3 -m pytest`

## Run Individual Research Experiments

Individual experiments are contained in the `research/**` folders.
- These are either bash scripts that launch sweeps across a SLURM cluster
- OR. individual python experiment scripts that can be run locally

Before this occurs, please run `bash research/e00_data_traversal/run_01_all_shared_data_prepare.sh`
to prepare all the needed data.

The working directory should always be the root of this project. Do not `cd` into
research directories when running experiments.

NOTE:
- Scipts may need to be modified to change [Weights and Biases](https://docs.wandb.ai/quickstart) user and project
  details. You will also need to login to wandb using their CLI.
- SLURM experiments need to be run on a specific slurm partition. Partition details can be adjusted by editing or
  adding to the hydra configuration group `experiment/config/run_location/*.yaml`
