# Pix2Pix

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-PyTorch-red)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/gans/LICENSE)

<p align="center">
  <img src="https://i.ibb.co/c1S0F6L/4-Pix2-Pix.jpg" width="500" height="350">
</p>

## Abstract (2018)
We investigate conditional adversarial networks as a general-purpose solution to image-to-image translation problems. These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. This makes it possible to apply the same generic approach to problems that traditionally would require very different loss formulations. We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks. Indeed, since the release of the pix2pix software associated with this paper, a large number of internet users (many of them artists) have posted their own experiments with our system, further demonstrating its wide applicability and ease of adoption without the need for parameter tweaking. As a community, we no longer hand-engineer our mapping functions, and this work suggests we can achieve reasonable results without hand-engineering our loss functions either.

## Models: Generator and Discriminator

Discriminator has a simple architecture of a couple of CNN layers. It is a PatchGAN where the output is $N×N$. Each cell grid is responsible for a patch of the original image.

<p align="center">
    <em> Table: Discriminator of Pix2Pix </em>
</p>

<div align="center">
  
|   |        **Layer**       | **Activation** | **Feature Map** |   **Size**   | **Kernel** | **Stride** | **Padding** |
|---|:----------------------:|:--------------:|:---------------:|:------------:|:----------:|------------|-------------|
|   |    INPUT (2 IMAGES)    |                |                 | 3⋅ 2×256×256 |            |            |             |
| 1 |         CONV 2D        |   Leaky ReLU   |        64       |  64×128×128  |      4     |      2     |      1      |
| 2 |         CONV 2D        |   Leaky ReLU   |       128       |   128×64×64  |      4     |      2     |      1      |
| 3 |      Batch NORM 2D     |                |                 |              |            |            |             |
| 4 |         CONV 2D        |   Leaky ReLU   |       256       |   256×32×32  |      4     |      2     |      1      |
| 5 |      Batch NORM 2D     |                |                 |              |            |            |             |
| 6 |         CONV 2D        |   Leaky ReLU   |       512       |   512×31×31  |      4     |      2     |      1      |
| 7 |      Batch NORM 2D     |                |                 |              |            |            |             |
| 8 |         CONV 2D        |                | 1               | 1×30×30      |      4     |      2     |      1      |
|   | OUTPUT (PROB. AS GRID) |                |                 |    1×30×30   |            |            |             |

</div>

Generator is a U-NET with skip connections (With skip connections: concatenations of UP1 with D7, UP2 with D6, …, UP6 with D2, UP7 with D1, that is why feature maps are symmetrically). In the first part, the image is down-sampled to  $512×1×1$ and in the second part, the latter is up-sampled back to the original size of $3×256×256$ with skip connections symmetrical put between the output of down layers and up layers. 

<p align="center">
    <em> Table: Generator of Pix2Pix (U-NET architecture) </em>
</p>

<div align="center">
  
|   |   **Name**   |    **Layer**   | **Activation** |  **Feature Map** |  **Size**  | **Kernel** | **Stride** | **Padding** |
|---|:------------:|:--------------:|:--------------:|:----------------:|:----------:|:----------:|:----------:|:-----------:|
|   |              |  INPUT (IMAGE) |                |                  |  3×256×256 |            |            |             |
| 1 |   _Initial_  |     CONV 2D    |   Leaky ReLU   |        64        | 64×128×128 |      4     |      2     |      1      |
| 2 |    _D1-D7_   |     CONV 2D    |   Leaky ReLU   | 64, 128, 512 (5) |   512×2×2  |      4     |      2     |      1      |
|   |      × 7     |  BATCH NORM 2D |                |                  |            |            |            |             |
| 3 | _Bottleneck_ |     CONV 2D    |      ReLU      |        512       |   512×1×1  |      4     |      2     |      1      |
| 4 |   _UP1-UP7_  |  CONV TRANS 2D |      ReLU      | 512 (5), 128, 64 | 64×128×128 |      4     |      2     |      1      |
|   |      × 7     |     CONV 2D    |                |                  |            |            |            |             |
| 5 |    _Final_   |  CONV TRANS 2D |      Tanh      |                  | 3×256× 256 |      4     |      2     |      1      |
|   |              | OUTPUT (IMAGE) |                |                  |            |            |            |             |

</div>

## Loss
For loss, BCE is used for both Generator and Discriminator. More specifically, to train the Discriminator, the BCE loss among real images and fake images as they are produced are summed. The generator uses the BCE loss of fake images produced with a L1 additional term. As a result, the input feature map has double the size in each block of up layers so concatenation can take place. The objective of the Discriminator and Generator (loss function) as a result are:
* Discriminator: $max [E(D(x)) - E(D(G(z)))]$
* Generator: $max [ E(D(G(z))) ] + λ_1 \cdot L1(x, D(z))($