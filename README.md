# Towards Composable Distributions of Latent Space Augmentations

This repository contains the implementation of the Latent Augmentation Variational Autoencoder (LAVAE) as described in the paper "Towards Composable Distributions of Latent Space Augmentations".

arXiv: https://arxiv.org/abs/2303.03462

## Overview

LAVAE is a novel approach for latent space image augmentation that allows for easy combination of multiple augmentations. The framework is based on the Variational Autoencoder (VAE) architecture and uses a new method for augmentation via linear transformation within the latent space itself.

Key features:
- Composable framework for latent space image augmentation
- Linear transformations in latent space to represent image augmentations
- Ability to transfer learned latent spaces to new sets of augmentations
- Improved control and geometric interpretability of the latent space

## Installation

```bash
git clone https://github.com/SunayBhat1/Latent_Augmentation_VAE.git
cd Latent_Augmentation_VAE
```
## Usage

To train an LAVAE model:
```bash
python run_LAVAE.py --experiment_name my_experiment --AUGS flip_lr flip_ud --AUGS_TRANSFER shear_x canny_edge
```

For a full list of command-line arguments and their descriptions, run:

```bash
python run_LAVAE.py --help
```

## File Structure

- `run_LAVAE.py`: Main script for training and evaluating LAVAE models
- `models.py`: Contains the LAVAE model architecture
- `utils.py`: Utility functions for data loading, visualization, etc.
- `train_utils.py`: Training and loss functions

## Results

The LAVAE model demonstrates:
- Effective latent space augmentations
- Composability of augmentations in latent space
- Ability to transfer learned latent spaces to new augmentations
- Improved interpretability of the latent space through 2D projections

For detailed results and visualizations, please refer to the paper and the `Results` folder in the repository.

## Citation

If you use this code in your research, please cite our paper:

```
@article{pooladzandi2023composabledistributionslatentspace,
      title={Towards Composable Distributions of Latent Space Augmentations}, 
      author={Omead Pooladzandi and Jeffrey Jiang and Sunay Bhat and Gregory Pottie},
      year={2023},
      eprint={2303.03462},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2303.03462}, 
}
```
