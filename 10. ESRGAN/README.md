# ESRGAN

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-PyTorch-red)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/gans/LICENSE)

<p align="center">
  <img src="https://i.ibb.co/YDZKWf0/7-SRGAN.png" width="300" height="200">
</p>

## Abstract (2018)
Abstract. The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating realistic textures during single image super-resolution. However, the hallucinated details are often accompanied with unpleasant artifacts. To further enhance the visual quality, we thoroughly study three key components of SRGAN network architecture, adversarial loss and perceptual loss, and improve each of them to derive an Enhanced SRGAN (ESRGAN). In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency and texture recovery. Beneting from these improvements, the proposed ESRGAN achieves consistently better visual quality with more realistic and natural textures than SRGAN and won the first place in the PIRM2018-SR Challenge.

## Models: Generator and Discriminator

The Discriminator of ESRGAN is exactly the same as SRGAN.

The architecture of ESRGAN's Generator has two modifications: 
* Remove all Batch Normalization layers from SRGAN
* Replace original block with the proposed Residual-in-Residual Dense Block (RRDB)

<p align="center">
  <img src="https://i.ibb.co/72ZgvNF/Picture2.png" width="300" height="200">
</p>

<p align="center">
    <em> Table: Generator of ESRGAN </em>
</p>

<div align="center">
  
|   |      **Name**     |                **Layer**                | **Activation** | **Input Shape** |
|---|:-----------------:|:---------------------------------------:|:--------------:|:---------------:|
|   |                   |              INPUT (IMAGE)              |                |     3×96×96     |
| 1 |     _Initial_     |                 CONV 2D                 |      PReLU     |                 |
| 2 | _Residual Blocks_ | Each RRDB block contains x3 RDB blocks, |                |                 |
|   |     _×23 RRDB_    |     each of which has 5x Conv Blocks    |                |                 |
| 3 |       _Conv_      |                 CONV 2D                 |                |                 |
| 4 |    _Up Samples_   |                 UPSAMPLE                |                |                 |
|   |        _×2_       |                 CONV 2D                 |                |                 |
| 5 |      _Final_      |                 CONV 2D                 |   Leaky ReLU   |                 |
| 6 |      _Final_      |                 CONV 2D                 |                |                 |
|   |                   |              OUTPUT (PROB.)             |                |     3×96×96     |

</div>

<p align="center">
    <em> Table: Discriminator of ESRGAN (same as SRGAN) </em>
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

## Loss

The loss for Discriminator is $loss_D = -E[ D(res_h) ] - E[ D(G(res_l)) ] + λ_GP \cdot GP$

The loss for Generator is $loss_G=loss_{L1}+loss_{VGG}+loss_{adversarial}$ where
* $loss_{L1} = 0.01 \cdot L1 = 0.01 \cdot L1[G(res_l), res_h]$
* $loss_{VGG} = VGG[ G(res_l), res_h)  ] = MSE[ (G(res_l), res_h ]$ of VGG-19
* $loss_{adversarial} = -0.005E[ D(G(res_l)) ]$