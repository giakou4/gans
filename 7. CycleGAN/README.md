# CycleGAN

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-PyTorch-red)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/gans/LICENSE)

<p align="center">
  <img src="https://i.ibb.co/mHpVJDh/5-Cycle-GAN.png"  width="500" height="350">
</p>

## Abstract (2020)
Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G: X->Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F:Y->X and introduce a cycle consistency loss to enforce F(G(X))->X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.

## Models: Generator and Discriminator

Discriminator of CycleGAN is similar to the Discriminator of Pix2Pix. The difference lies that the input is 1 image rather than 2, so no doubling of the input channels occurs. Moreover, the normalization is Instance Normalization and not Batch Normalization. Moreover, the output passes through a Sigmoid Function.

<p align="center">
    <em> Table: Discriminator of CycleGAN </em>
</p>

<div align="center">
  
|   |   **Name**  |     **Layer**    | **Activation** | **Feature Map** |  **Size**  | **Kernel** | **Stride** | **Padding** |
|---|:-----------:|:----------------:|:--------------:|:---------------:|:----------:|:----------:|:----------:|:-----------:|
|   |             |   INPUT (IMAGE)  |                |                 |  3×256×256 |            |            |             |
| 1 |             |      CONV 2D     |      ReLU      |                 | 64×256×256 |      7     |      1     |      3      |
| 2 |             | INSTANCE NORM 2D |                |                 |            |            |            |             |
| 3 |   _D1-D2_   |      CONV 2D     |      ReLU      |     128, 256    |   512×2×2  |      3     |      2     |      1      |
|   |     ×7      | INSTANCE NORM 2D |                |                 |            |            |            |             |
| 4 | _Res1-Res9_ |      CONV 2D     |      ReLU      |    256 (all)    |            |      3     |      1     |      1      |
|   |     ×9      | INSTANCE NORM 2D |                |                 |            |            |            |             |
|   |             |      CONV 2D     |                |                 |  256×64×64 |      3     |      1     |      1      |
|   |             | INSTANCE NORM 2D |                |                 |            |            |            |             |
| 5 |  _UP1-UP2_  |   CONV TRANS 2D  |      ReLU      |     128, 64     | 64×256×256 |      3     |      2     |      1      |
|   |     ×2      | INSTANCE NORM 2D |                |                 |            |            |            |             |
| 6 |             |      CONV 2D     |      Tanh      |        3        |            |            |            |             |
|   |             |  OUTPUT (IMAGE)  |                |                 |            |            |            |             |

</div>

For the Generator a ResNet like implementation is done. In the residual blocks, the output is $out=x+Net(x)$, where $Net$ represents both CONV 2D and INSTANCE NORM 2D of the residual block.

<p align="center">
    <em> Table: Generator of CycleGAN (ResNet) </em>
</p>

<div align="center">
  
|   |  **Name** |     **Layer**    | **Activation** | **Feature Map** |  **Size**  | **Kernel** | **Stride** | **Padding** |
|---|:---------:|:----------------:|:--------------:|:---------------:|:----------:|:----------:|:----------:|:-----------:|
|   |           |   INPUT (IMAGE)  |                |                 |  3×256×256 |            |            |             |
| 1 |           |      CONV 2D     |      ReLU      |                 | 64×256×256 |      7     |      1     |      3      |
| 2 |           | INSTANCE NORM 2D |                |                 |            |            |            |             |
| 3 |   D1-D2   |      CONV 2D     |      ReLU      |     128, 256    |   512×2×2  |      3     |      2     |      1      |
|   |     ×7    | INSTANCE NORM 2D |                |                 |            |            |            |             |
| 4 | Res1-Res9 |      CONV 2D     |      ReLU      |    256 (all)    |            |      3     |      1     |      1      |
|   |     ×9    | INSTANCE NORM 2D |                |                 |            |            |            |             |
|   |           |      CONV 2D     |                |                 |  256×64×64 |      3     |      1     |      1      |
|   |           | INSTANCE NORM 2D |                |                 |            |            |            |             |
| 5 |  UP1-UP2  |   CONV TRANS 2D  |      ReLU      |     128, 64     | 64×256×256 |      3     |      2     |      1      |
|   |     ×2    | INSTANCE NORM 2D |                |                 |            |            |            |             |
| 6 |           |      CONV 2D     |      Tanh      |        3        |            |            |            |             |
|   |           |  OUTPUT (IMAGE)  |                |                 |            |            |            |             |

</div>

## Loss
Given that we have:
* Two mappings (Generators): $G:X→Y $and F$:Y→X$
* Two Discriminators: $D_X$ and $D_Y$ where $D_X$ aims to discriminate between ${x}$ and ${F(y)}$ and $D_Y$ between ${y}$ and ${G(x)}$  
* A Generator $G$ that tries to generate images $G(x)$ that look similar to images from domain $Y$

The losses are 2 adversarial and 1 cycle:
* $min$  $max [L_{GAN} (G, D_Y, X, Y) ]$ where the latter is an MSE loss $L_{GAN} = E[log(D_Y(y)] + E[ log(1-D_Y(G(x))) ]$
* $min$ $max [L_{GAN} (G, D_X, X, Y) $
* $L_{cycle}=E[ F(G(x)) - x ] + E[ G(F(x))-y ]$ where the latter is L1 loss