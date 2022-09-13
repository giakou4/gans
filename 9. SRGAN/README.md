# SRGAN

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-PyTorch-red)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/gans/LICENSE)

<p align="center">
  <img src="https://i.ibb.co/YDZKWf0/7-SRGAN.png" width="300" height="200">
</p>

## Abstract (2017)
Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains
largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this paper, we present SRGAN, a generative adversarial network (GAN) for image superresolution (SR). To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4x upscaling factors. To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks. An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN. The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art method.

## Models: Generator and Discriminator

The architecture of Generator and Discriminator networks is as follows:

<p align="center">
  <img src="https://i.ibb.co/fFs7KkS/Picture1.png" width="800" height="500">
</p>

<p align="center">
    <em> Table: Discriminator of SRGAN </em>
</p>

<div align="center">
  
|   |   **Name**   |     **Layer**    | **Activation** |  **Input Shape**  | **Kernel** | **Stride** | **Padding** |
|---|:------------:|:----------------:|:--------------:|:-----------------:|:----------:|:----------:|:-----------:|
|   |              |   INPUT (IMAGE)  |                |      3×96×96      |            |            |             |
| 1 |   _Blocks_   |      CONV 2D     |   Leaky ReLU   | based on features |      3     |      1     |      1      |
|   |      ×8      |    BATCH NORM    |                |                   |            |            |             |
| 2 | _Classifier_ | ADAPTIVE POOL 2D |                |      512×6×6      |      3     |      1     |      1      |
|   |              |      FLATTEN     |                |       18.432      |            |            |             |
|   |              |      LINEAR      |   Leaky ReLU   |       1.024       |            |            |             |
|   |              |      LINEAR      |                |         1         |            |            |             |
|   |              |  OUTPUT (PROB.)  |                |         1         |            |            |             |

</div>

<p align="center">
    <em> Table: Generator of SRGAN </em>
</p>

<div align="center">
  
|   |   **Name**   |    **Layer**   | **Activation** | **Input Shape** | **Kernel** | **Stride** | **Padding** |
|---|:------------:|:--------------:|:--------------:|:---------------:|:----------:|:----------:|:-----------:|
|   |              |  INPUT (IMAGE) |                |     3×24×24     |            |            |             |
| 1 |   _Initial_  |     CONV 2D    |      PReLU     |     64×24×24    |      9     |      1     |      4      |
| 2 |  _Residual_  |     CONV 2D    |      PReLU     |     64×24×24    |      3     |      1     |      1      |
|   |    _Block_   |   BATCH NORM   |                |                 |            |            |             |
|   |      ×16     |     CONV 2D    |      PReLU     |     64×24×24    |      3     |      1     |      1      |
|   |              |   BATCH NORM   |                |                 |            |            |             |
| 3 | _Conv Block_ |     CONV 2D    |                |     64×24×24    |      3     |      1     |      1      |
| 4 |              |   BATCH NORM   |                |                 |            |            |             |
| 5 |     _Up_     |     CONV 2D    |      PReLU     |     64×48×48    |      3     |      1     |      1      |
|   |   _Samples_  |  PIXEL SHUFFLE |                |                 |            |            |             |
|   |      ×2      |     CONV 2D    |      PReLU     |     64×96×96    |      3     |      1     |      1      |
|   |              |  PIXEL SHUFFLE |                |                 |            |            |             |
| 6 |    _Final_   |     CONV 2D    |      Tanh      |     3×96×96     |      9     |      1     |      4      |
|   |              | OUTPUT (IMAGE) |                |     3×96×96     |            |            |             |

</div>

## Loss
The objective of the Discriminator and Generator are:
* Discriminator: $max [ log(D(x)) + log( 1-D(G(z)) ) ]$
* Generator: $min [ log(1-D(G(z)) ]$ or $max[ log(D(G(z)) ]$ 

The loss of the Discriminator and Generator are:
* Discriminator: $loss_D = BCE[ D(res_h), 1-0.01 \cdot D(res_h) ] + BCE[ D(G(res_l)), 0 ]$
* Generator: $loss_G = loss_{adversarial} + loss_{VGG} = 0.001 \cdot BCE[ D(G(res_l)), 1 )]+ 0.006 \cdot VGG[ G(res_l), res_h ]$

where $res_l$ is the low resolution image of size $3×24×24$ and $res_h$ is the high resolution image of size $3×96×96$.
