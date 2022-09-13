# Simple GAN

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-PyTorch-red)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/gans/LICENSE)

GANs consist of 2 networks playing an adversarial game against each other: a Generator (counterfeiter) and a Discriminator (detective). In the end, the Generator generates indistinguishable fake images from real ones and the Discriminator is forced to guess with probability 1/2. Both Generator and Discriminator are randomly initialized and simultaneously trained. 

<div align="center">
  
|            | **Generator** |    **Discriminator**    |
|------------|:-------------:|:-----------------------:|
| **Input**  |     noise     |          image          |
| **Output** |     image     | probability (real/fake) |

</div>

## Generator and Discriminator
A simple Generator and Discriminator are as follows

<p align="center">
    <em> Table: Discriminator of a Simple GAN </em>
</p>

<div align="center">

|   |       **Layer**       | **Activation** | **Feature Map** | **Size** |
|---|:---------------------:|:--------------:|:---------------:|:--------:|
|   | INPUT (FLATTEN IMAGE) |                |                 |   1×784  |
| 1 |         LINEAR        |   Leaky ReLU   |       128       |   1×128  |
| 2 |         LINEAR        |      Tanh      |        1        |    1×1   |
|   |  OUTPUT (RPOBABILITY) |                |                 |     1    |

</div>

<p align="center">
    <em> Table: Generator of a Simple GAN </em>
</p>

<div align="center">

|   |       **Layer**       | **Activation** | **Feature Map** | **Size** |
|---|:---------------------:|:--------------:|:---------------:|:--------:|
|   | INPUT (FLATTEN IMAGE) |                |                 |   1×64   |
| 1 |         LINEAR        |   Leaky ReLU   |       256       |   1×256  |
| 2 |         LINEAR        |      Tanh      |       784       |   1×784  |
|   |  OUTPUT (RPOBABILITY) |                |                 |   1×784  |

</div>