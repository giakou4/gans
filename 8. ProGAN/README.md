# ProGAN 

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-PyTorch-red)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/gans/LICENSE)

<p align="center">
  <img src="https://i.ibb.co/LSH12Jk/Screenshot-1.png" width="800" height="200">
</p>

## Abstract (2018)
We describe a new training methodology for generative adversarial networks. The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality, e.g., CELEBA images at 10242. We also propose a simple way to increase the variation in generated images, and achieve a record inception score of 8:80 in unsupervised CIFAR10. Additionally, we describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator. Finally, we suggest a new metric for evaluating GAN results, both in terms of image quality and variation. As an additional contribution, we construct a higher-quality version of the CELEBA dataset.

## Innovations
* Grow Generator and Discriminator progressively
* Mini Batch std       
* Pixel Norm: Normalize each pixel value $a_(x,y)$ by the mean of pixel squared: $b_(x,y)=a_(x,y)/E[a_(x,y)]$
* Equalize Learning Rate $W_f=W_i \cdot f(k, c)$ (He-initialization)

## Models: Generator and Discriminator

Both Generator and Discriminator are hard to implement.

<p align="center">
    <em> Table: Discriminator of ProGAN </em>
</p>

<div align="center">
  
|    |    **Layer**   | **Activation** |   **Size**   | **Kernel** | **Stride** | **Padding** |
|----|:--------------:|:--------------:|:------------:|:----------:|:----------:|:-----------:|
|    |  INPUT (IMAGE) |                |  3×1024×1024 |            |            |             |
| 1  |     CONV 2D    |  Leaky ReLU    | 16×1024×1024 |      4     |      0     |      0      |
| 2  |   Conv Block   |  Leaky ReLU    | 32×1024×1024 |      3     |      1     |      1      |
| 3  |      DOWN      |                |  32×512×512  |            |            |             |
| 4  |   Conv Block   |  Leaky ReLU    |  64×512×512  |      3     |      1     |      1      |
| 5  |      DOWN      |                |  64×256×256  |            |            |             |
| 6  |   Conv Block   |  Leaky ReLU    |  128×256×256 |      3     |      1     |      1      |
| 7  |      DOWN      |                |  128×128×128 |            |            |             |
| 8  |   Conv Block   |  Leaky ReLU    |  256×128×128 |      3     |      1     |      1      |
| 9  |      DOWN      |                |   256×64×64  |            |            |             |
| 10 |   Conv Block   |  Leaky ReLU    |   512×64×64  |      3     |      1     |      1      |
| 11 |      DOWN      |                |   512×32×32  |            |            |             |
| 12 |   Conv Block   |  Leaky ReLU    |   512×32×32  |      3     |      1     |      1      |
| 13 |      DOWN      |                |   512×16×16  |            |            |             |
| 14 |   Conv Block   |  Leaky ReLU    |   512×16×16  |      3     |      1     |      1      |
| 15 |      DOWN      |                |    512×8×8   |            |            |             |
| 16 |   Conv Block   |  Leaky ReLU    |    512×8×8   |      3     |      1     |      1      |
| 17 |      DOWN      |                |    512×4×4   |            |            |             |
| 19 |   Conv Block   |  Leaky ReLU    |    512×4×4   |      3     |      1     |      1      |
| 19 |     CONV 2D    |                |    512×1×1   |      4     |      4     |      0      |
| 20 |       FC       |     Linear     |     1×1×1    |            |            |             |
|    | OUTPUT (PROB.) |                |       1      |            |            |             |

</div>

<p align="center">
    <em> Table: Generator of ProGAN </em>
</p>

<div align="center">
  
|    |    **Layer**   | **Activation** |    **Size**   | **Kernel** | **Stride** | **Padding** |
|----|:--------------:|:--------------:|:-------------:|:----------:|:----------:|:-----------:|
|    |  INPUT (NOISE) |                |    512×1×1    |            |            |             |
| 1  |     CONV 2D    |   Leaky ReLU   |    512×4×4    |      4     |      4     |      0      |
| 2  |   Conv Block   |   Leaky ReLU   |    512×4×4    |      3     |      1     |      1      |
| 3  |    UPSAMPLE    |                |    512×8×8    |            |            |             |
| 4  |   Conv Block   |   Leaky ReLU   |    512×4×4    |      3     |      1     |      1      |
| 5  |    UPSAMPLE    |                |   512×16×16   |            |            |             |
| 6  |   Conv Block   |   Leaky ReLU   |   512×16×16   |      3     |      1     |      1      |
| 7  |    UPSAMPLE    |                |   512×32×32   |            |            |             |
| 8  |   Conv Block   |   Leaky ReLU   |   512×32×32   |      3     |      1     |      1      |
| 9  |    UPSAMPLE    |                |   512×64×64   |            |            |             |
| 10 |   Conv Block   |   Leaky ReLU   |   512×64×64   |      3     |      1     |      1      |
| 11 |    UPSAMPLE    |                |  512×128×128  |            |            |             |
| 12 |   Conv Block   |   Leaky ReLU   |  512×128×128  |      3     |      1     |      1      |
| 13 |    UPSAMPLE    |                |  512×256×256  |            |            |             |
| 14 |   Conv Block   |   Leaky ReLU   |  512×256×256  |      3     |      1     |      1      |
| 15 |    UPSAMPLE    |                |  512×512×512  |            |            |             |
| 16 |   Conv Block   |   Leaky ReLU   |  512×512×512  |      3     |      1     |      1      |
| 17 |    UPSAMPLE    |                | 512×1024×1024 |            |            |             |
| 19 |   Conv Block   |   Leaky ReLU   | 512×1024×1024 |      3     |      1     |      1      |
| 19 |   WS CONV 2D   |     Linear     |  3×1024×1024  |            |            |             |
|    | OUTPUT (PROB.) |                |  3×1024×1024  |            |            |             |

</div>

## Loss
The loss for Discriminator is $loss_D = -D(x, a-D(G(z), a) + λ_GP \cdot GP+0.001 \cdot D(x,a)^2$. 

The loss for Generator is $loss_G = -D(G(z),a)$
