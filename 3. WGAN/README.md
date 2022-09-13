# WGAN

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-PyTorch-red)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/gans/LICENSE)

<p align="center">
  <img src="https://i.ibb.co/BBtZ2Wc/2-WGAN.png" width="300" height="200">
</p>

## Abstract (2017)
We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions.

## Generator and Critic

Models now includes Batch Normalization. Discriminator does not have Sigmoid function, as a result it is called Critic.

<p align="center">
    <em> Table: Critic of WGAN </em>
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
    <em> Table: Generator of WGAN </em>
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

## Loss

We want the distribution of generator $P_g$ and of the real images $P_r$ to be similar. How to define distance between distributions (e.g., Kullback-Leibler (KL) divergence, Jensen-Shannon (JS) divergence, Wasserstein Distance)?

$$max(E_{x \to P_r} [D(x)]-E_{x \to P_g}[D(x)])$$

Discriminator wants to maximize the above equation, while Generator was to minimize. Converge when it’s close to 0. Hence, loss means something!

The objective of the Critic and Generator (loss function) as a result are:
* Critic: $max [E(C(x)) - E(C(G(z)))]$
* Generator: $max [ E(C(G(z))) ]$ or $min[-E(C(G(z))]$