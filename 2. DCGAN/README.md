# DCGAN

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-PyTorch-red)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/gans/LICENSE)

<p align="center">
  <img src="https://i.ibb.co/F5WKZxX/1-Simple-GAN.png" width="300" height="200">
</p>

## Abstract (2016)
In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generativeadversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations

## Generator and Discriminator
Models include Convolutional Neural Networks (CNN) since images are used. Discriminator uses Convolutional layers while Generator uses Transpose Convolutional layers. The output of Discriminator passes through a Sigmoid activation function since it represents probability (fake or real), while the output of Generators though a Tanh, to assure the output is an image and is within [-1,1].

<p align="center">
    <em> Table: Discriminator of DCGAN</em>
</p>

<div align="center">

|            |       Layer      |     Activation    | Feature Map |    Size   | Kernel | Stride | Padding |
|------------|:----------------:|:-----------------:|:-----------:|:---------:|:------:|:------:|:-------:|
|            |   INPUT (IMAGE)  |                   |             |  3x64x64  |        |        |         |
|     1      |      CONV 2D     |     Leaky ReLU    |      64     |  64×32×32 |    4   |    2   |    1    |
|     2      |      CONV 2D     |     Leaky ReLU    |     128     | 128×16×16 |    4   |    2   |    1    |
|     3      |      CONV 2D     |     Leaky ReLU    |     256     |  256×8×8  |    4   |    2   |    1    |
|     4      |      CONV 2D     |     Leaky ReLU    |     512     |  512×4×4  |    4   |    2   |    1    |
|     5      |      CONV 2D     |       Sigmoid     |      1      |   1×1×1   |    4   |    2   |    1    |
|            |  OUTPUT (PROB.)  |                   |             |     1     |        |        |         |

</div>

<p align="center">
    <em> Table: Generator of DCGAN </em>
</p>

<div align="center">

|            |          Layer         |     Activation    | Feature Map |    Size   | Kernel | Stride | Padding |
|------------|:----------------------:|:-----------------:|:-----------:|:---------:|:------:|:------:|:-------:|
|            |      INPUT (NOISE)     |                   |             |  100×1× 1 |        |        |         |
|     1      |    CONV TRANSPOSE 2D   |        ReLU       |     1024    |  1024×4×4 |    4   |    1   |    0    |
|     2      |    CONV TRANSPOSE 2D   |        ReLU       |     512     |  512×8×8  |    4   |    2   |    0    |
|     3      |    CONV TRANSPOSE 2D   |        ReLU       |     256     | 256×16×16 |    4   |    2   |    0    |
|     4      |    CONV TRANSPOSE 2D   |        ReLU       |     128     | 128×32×32 |    4   |    2   |    0    |
|     5      |    CONV TRANSPOSE 2D   |        Tanh       |      3      |  3×64×64  |    4   |    2   |    0    |
|            |     OUTPUT (IMAGE)     |                   |             |  3x64x64  |        |        |         |

</div>

## Loss
The objective of the Discriminator and Generator (loss function) are:
* Discriminator: $max [E(log(D(x)+log(1-D(G(z)))]$
* Generator: $min[E(log(1-D(G(z))))]$ or $max[log(D(G(z)))]$

where they both can be expressed as
$$min_{G}max_{D}V(D,G)=E[log(D(x)] + E[log(1-D(G(z))]$$
