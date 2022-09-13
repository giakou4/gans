# WGAN-GP (with Gradient Penalty)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-PyTorch-red)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/gans/LICENSE)

<p align="center">
  <img src="https://i.ibb.co/BBtZ2Wc/2-WGAN.png" width="300" height="200">
</p>

A new paper introduced the WGAN with Gradient Penalty (GP), where in the loss function of Critic, a penalty is added. 

## Abstract (2017)
Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The recently proposedWasserstein GAN (WGAN) makes progress toward stable training of GANs, but sometimes can still generate only poor samples or fail to converge. We find that these problems are often due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior. We propose an alternative to clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no hyperparameter tuning, including 101-layer ResNets and language models with continuous generators. We also achieve high quality generations on CIFAR-10 and LSUN bedrooms.

## Generator, Critic and Loss
WGAN and WGAN-GP use the exact same models for Generator and Critic. The only difference lies is the loss back propagated, where a penalty with a weight is added $λ_{GP} \cdot GP$:
* Critic: $max [E(C(x)) - E(C(G(z))) ] + λ_{GP} \cdot GP$
* Generator: $max [ E(C(G(z))) ]$ or $min[-E(C(G(z))]$

<p align="center">
    <em> Table: Critic of WCGAN-GP and WGAN </em>
</p>

<div align="center">
  
|   |     **Layer**    | **Activation** | **Feature Map** |  **Size** | **Kernel** | **Stride** | **Padding** |
|---|:----------------:|:--------------:|:---------------:|:---------:|:----------:|------------|-------------|
|   |   INPUT (IMAGE)  |                |                 |  3×64×64  |            |            |             |
| 1 |      CONV 2D     |   Leaky ReLU   |        64       |  64×32×32 |      4     |      2     |      1      |
| 2 |      CONV 2D     |   Leaky ReLU   |       128       | 128×16×16 |      4     |      2     |      1      |
| 3 | INSTANCE NORM 2D |                |                 |           |            |            |             |
| 4 |      CONV 2D     |   Leaky ReLU   |       256       |  256×8×8  |      4     |      2     |      1      |
| 5 | INSTANCE NORM 2D |                |                 |           |            |            |             |
| 6 |      CONV 2D     |   Leaky ReLU   |       512       |  512×4×4  |      4     |      2     |      1      |
| 7 | INSTANCE NORM 2D |                |                 |           |            |            |             |
| 8 |      CONV 2D     |                |        1        |   1×1×1   |      4     |      2     |      0      |
|   |  OUTPUT (PROB.)  |                |                 |     1     |            |            |             |

</div>

<p align="center">
    <em> Table: Generator of WGAN-GP and WCGAN </em>
</p>

<div align="center">
  
|   |    **Layer**   | **Activation** | **Feature Map** |  **Size** | **Kernel** | **Stride** | **Padding** |
|---|:--------------:|:--------------:|:---------------:|:---------:|:----------:|------------|-------------|
|   |  INPUT (NOISE) |                |                 |  100×1×1  |            |            |             |
| 1 |  CONV TRANS 2D |      ReLU      |       1024      |  1024×4×4 |            |            |             |
| 2 |  BATCH NORM 2D |                |                 |           |            |            |             |
| 3 |  CONV TRANS 2D |      ReLU      |       512       |  512×8×8  |            |            |             |
| 4 |  BATCH NORM 2D |                |                 |           |            |            |             |
| 5 |  CONV TRANS 2D |      ReLU      |       256       | 256×16×16 |            |            |             |
| 6 |  BATCH NORM 2D |                |                 |           |            |            |             |
| 7 |  CONV TRANS 2D |      ReLU      |       128       | 128×32×32 |            |            |             |
| 8 |  BATCH NORM 2D |                |                 |           |            |            |             |
| 9 |  CONV TRANS 2D |      Tanh      |        3        |  3×64×64  |            |            |             |
|   | OUTPUT (IMAGE) |                |                 |     1     |            |            |             |

</div>