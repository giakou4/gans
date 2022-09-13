# Generative Adversarial Networks (GANs)

Implementation of GAN architectures in [PyTorch](https://pytorch.org/)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-PyTorch-red)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/gans/LICENSE)

The structure of code is as follows:  
```bash
gan/  
├── logs
├── checkpoints
│   ├── disc.pth.tar  
│   └── gen.pth.tar  
├── data
│   ├── MNIST
├── train.py  
├── model.py 
├── dataset.py 
├── utils.py 
└── README.md
```

Each _model.py_ has the two following class implementations: 
```python 
class Discriminator(torch.nn.Module):
    """ Discriminator of XXX paper """
    def __init__(self, img_channels=3):
        pass
    def forward(self, x)
        return x

class Generator(torch.nn.Module):
    """ Generator of XXX paper """
    def __init__(self, img_channels=3, noise_dim=512):
        pass
    def forward(self, x)
        return x
```

Each _train.py_ has an arguement parser, a function for single epoch training and the main function 
```python
def parse_opt():
    parser = argparse.ArgumentParser()
    # ...
    opt = parser.parse_args()
    return opt

def train_one_epoch(loader, gen, disc, opt_gen, opt_disc, loss, g_scaler, d_scaler, writer, tb_step, epoch, num_epochs, **kwargs):
  pass

def main(config):
  pass
  
if __name__ == "__main__":
    config = prase_opt()
    main(config)
```

In the _dataset.py_ we define, unless PyTorchs ```ImageFolder``` is fine, the
```python
class MyImageFolder(torch.utils.data.Dataset):
    """ My image dataset """
    pass
```


It the _utils.py_, we define two basic functions: ```save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar")``` and ```load_checkpoint(checkpoint_file, model, optimizer, lr, device)``` among other essential for training.

## 1. Simple GAN
GANs consist of 2 networks playing an adversarial game against each other: a Generator (counterfeiter) and a Discriminator (detective). In the end, the Generator generates indistinguishable fake images from real ones and the Discriminator is forced to guess with probability 1/2. Both Generator and Discriminator are randomly initialized and simultaneously trained. 

<div align="center">
  
|            | **Generator** |    **Discriminator**    |
|------------|:-------------:|:-----------------------:|
| **Input**  |     noise     |          image          |
| **Output** |     image     | probability (real/fake) |

</div>

### Generator and Discriminator
A simple Generator and Discriminator are as follows

<p align="center">
    <em> Table 1: Discriminator of a Simple GAN </em>
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
    <em> Table 2: Generator of a Simple GAN </em>
</p>

<div align="center">

|   |       **Layer**       | **Activation** | **Feature Map** | **Size** |
|---|:---------------------:|:--------------:|:---------------:|:--------:|
|   | INPUT (FLATTEN IMAGE) |                |                 |   1×64   |
| 1 |         LINEAR        |   Leaky ReLU   |       256       |   1×256  |
| 2 |         LINEAR        |      Tanh      |       784       |   1×784  |
|   |  OUTPUT (RPOBABILITY) |                |                 |   1×784  |

</div>

## 2. DCGAN
<p align="center">
  <img src="https://i.ibb.co/F5WKZxX/1-Simple-GAN.png" width="300" height="200">
</p>

### 2.1 Abstract (2016)
In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generativeadversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations

### 2.2 Generator and Discriminator
Models include Convolutional Neural Networks (CNN) since images are used. Discriminator uses Convolutional layers while Generator uses Transpose Convolutional layers. The output of Discriminator passes through a Sigmoid activation function since it represents probability (fake or real), while the output of Generators though a Tanh, to assure the output is an image and is within [-1,1].

<p align="center">
    <em> Table 3: Discriminator of DCGAN</em>
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
    <em> Table 4: Generator of DCGAN </em>
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
|            |     OUTPUT (IMAGE)     |                   |             |     1     |        |        |         |

</div>

### 2.3 Loss
The objective of the Discriminator and Generator (loss function) are:
* Discriminator: $max [E(log(D(x)+log(1-D(G(z)))]$
* Generator: $min[E(log(1-D(G(z))))]$ or $max[log(D(G(z)))]$

where they both can be expressed as
$$min_{G}max_{D}V(D,G)=E[log(D(x)] + E[log(1-D(G(z))]$$

## 3. WGAN
<p align="center">
  <img src="https://i.ibb.co/BBtZ2Wc/2-WGAN.png" width="300" height="200">
</p>

### 3.1 Abstract (2017)
We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions.

### 3.2 Generator and Critic

Models now includes Batch Normalization. Discriminator does not have Sigmoid function, as a result it is called Critic.

<p align="center">
    <em> Table 5: Critic of WGAN </em>
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
    <em> Table 6: Generator of WGAN </em>
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

### 3.3 Loss

We want the distribution of generator $P_g$ and of the real images $P_r$ to be similar. How to define distance between distributions (e.g., Kullback-Leibler (KL) divergence, Jensen-Shannon (JS) divergence, Wasserstein Distance)?

$$max(E_{x \to P_r} [D(x)]-E_{x \to P_g}[D(x)])$$

Discriminator wants to maximize the above equation, while Generator was to minimize. Converge when it’s close to 0. Hence, loss means something!

The objective of the Critic and Generator (loss function) as a result are:
* Critic: $max [E(C(x)) - E(C(G(z)))]$
* Generator: $max [ E(C(G(z))) ]$ or $min[-E(C(G(z))]$

## 4. WGAN-GP (with Gradient Penalty)
<p align="center">
  <img src="https://i.ibb.co/BBtZ2Wc/2-WGAN.png" width="300" height="200">
</p>

A new paper introduced the WGAN with Gradient Penalty (GP), where in the loss function of Critic, a penalty is added. 

### 4.1 Abstract (2017)
Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The recently proposedWasserstein GAN (WGAN) makes progress toward stable training of GANs, but sometimes can still generate only poor samples or fail to converge. We find that these problems are often due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior. We propose an alternative to clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no hyperparameter tuning, including 101-layer ResNets and language models with continuous generators. We also achieve high quality generations on CIFAR-10 and LSUN bedrooms.

### 4.2 Generator, Critic and Loss
WGAN and WGAN-GP use the exact same models for Generator and Critic. The only difference lies is the loss back propagated, where a penalty with a weight is added $λ_{GP} \cdot GP$:
* Critic: $max [E(C(x)) - E(C(G(z))) ] + λ_{GP} \cdot GP$
* Generator: $max [ E(C(G(z))) ]$ or $min[-E(C(G(z))]$

## 5. Conditional GAN
<p align="center">
  <img src="https://i.ibb.co/JkHZqvY/3-Conditional-GAN.png" width="300" height="200">
</p>

### 5.1 Abstract (2017)
Generative Adversarial Nets were recently introduced as a novel way to train generative models. In this work we introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data, y, we wish to condition on to both the generator and discriminator. We show that this model can generate MNIST digits conditioned on class labels. We also illustrate how this model could be used to learn a multi-modal model, and provide preliminary examples of an application to image tagging in which we demonstrate how this approach can generate descriptive tags which are not part of training labels.

### 5.2 Generator and Critic
Conditional GANs have information about the label, so both Discriminator and Generator are supervised. The Generator generates what you want him to generate. It can be seen as an extension of WGAN. Discriminator includes an embedding from number of classes $num-classes$ to the image’s size squared $(img-size)^2$. The labels are embedded in that layer and resized to $img-size×img-size$. As a result, the first CONV 2D layer has 3+1 channels. Generator also includes an embedding. The labels are embedded and unsqueezed twice producing an extra channel. As a result, the first CONV 2D layer has 3+1 channels.

## 6. Pix2Pix
<p align="center">
  <img src="https://i.ibb.co/c1S0F6L/4-Pix2-Pix.jpg" width="500" height="350">
</p>

### 6.1 Abstract (2018)
We investigate conditional adversarial networks as a general-purpose solution to image-to-image translation problems. These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. This makes it possible to apply the same generic approach to problems that traditionally would require very different loss formulations. We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks. Indeed, since the release of the pix2pix software associated with this paper, a large number of internet users (many of them artists) have posted their own experiments with our system, further demonstrating its wide applicability and ease of adoption without the need for parameter tweaking. As a community, we no longer hand-engineer our mapping functions, and this work suggests we can achieve reasonable results without hand-engineering our loss functions either.

### 6.2 Models: Generator and Discriminator

Discriminator has a simple architecture of a couple of CNN layers. It is a PatchGAN where the output is $N×N$. Each cell grid is responsible for a patch of the original image.

<p align="center">
    <em> Table 7: Discriminator of Pix2Pix </em>
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
    <em> Table 8: Generator of Pix2Pix (U-NET architecture) </em>
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

### 6.3 Loss
For loss, BCE is used for both Generator and Discriminator. More specifically, to train the Discriminator, the BCE loss among real images and fake images as they are produced are summed. The generator uses the BCE loss of fake images produced with a L1 additional term. As a result, the input feature map has double the size in each block of up layers so concatenation can take place. The objective of the Discriminator and Generator (loss function) as a result are:
* Discriminator: $max [E(D(x)) - E(D(G(z)))]$
* Generator: $max [ E(D(G(z))) ] + λ_1 \cdot L1(x, D(z))($

## 7. CycleGAN
<p align="center">
  <img src="https://i.ibb.co/mHpVJDh/5-Cycle-GAN.png"  width="500" height="350">
</p>

### 7.1 Abstract (2020)
Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G: X->Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F:Y->X and introduce a cycle consistency loss to enforce F(G(X))->X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.

### 7.2 Models: Generator and Discriminator

Discriminator of CycleGAN is similar to the Discriminator of Pix2Pix. The difference lies that the input is 1 image rather than 2, so no doubling of the input channels occurs. Moreover, the normalization is Instance Normalization and not Batch Normalization. Moreover, the output passes through a Sigmoid Function.

<p align="center">
    <em> Table 9: Discriminator of CycleGAN </em>
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
    <em> Table 10: Generator of CycleGAN (ResNet) </em>
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

### 7.3 Loss
Given that we have:
* Two mappings (Generators): $G:X→Y $and F$:Y→X$
* Two Discriminators: $D_X$ and $D_Y$ where $D_X$ aims to discriminate between ${x}$ and ${F(y)}$ and $D_Y$ between ${y}$ and ${G(x)}$  
* A Generator $G$ that tries to generate images $G(x)$ that look similar to images from domain $Y$

The losses are 2 adversarial and 1 cycle:
* $min$  $max [L_{GAN} (G, D_Y, X, Y) ]$ where the latter is an MSE loss $L_{GAN} = E[log(D_Y(y)] + E[ log(1-D_Y(G(x))) ]$
* $min$ $max [L_{GAN} (G, D_X, X, Y) $
* $L_{cycle}=E[ F(G(x)) - x ] + E[ G(F(x))-y ]$ where the latter is L1 loss

## 8. ProGAN 
<p align="center">
  <img src="https://i.ibb.co/LSH12Jk/Screenshot-1.png" width="800" height="200">
</p>

### 8.1 Abstract (2018)
We describe a new training methodology for generative adversarial networks. The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality, e.g., CELEBA images at 10242. We also propose a simple way to increase the variation in generated images, and achieve a record inception score of 8:80 in unsupervised CIFAR10. Additionally, we describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator. Finally, we suggest a new metric for evaluating GAN results, both in terms of image quality and variation. As an additional contribution, we construct a higher-quality version of the CELEBA dataset.

### 8.2 Innovations
* Grow Generator and Discriminator progressively
* Mini Batch std       
* Pixel Norm: Normalize each pixel value $a_(x,y)$ by the mean of pixel squared: $b_(x,y)=a_(x,y)/E[a_(x,y)]$
* Equalize Learning Rate $W_f=W_i \cdot f(k, c)$ (He-initialization)

### 8.3 Models: Generator and Discriminator

Both Generator and Discriminator are hard to implement.

<p align="center">
    <em> Table 11: Discriminator of ProGAN </em>
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
    <em> Table 12: Generator of ProGAN </em>
</p>

<div align="center">
  
|    |    **Layer**   | **Activation** |    **Size**   | **Kernel** | **Stride** | **Padding** |
|----|:--------------:|:--------------:|:-------------:|:----------:|:----------:|:-----------:|
|    |  INPUT (IMAGE) |                |    512×1×1    |            |            |             |
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

### 8.4 Loss
The loss for Discriminator is $loss_D = -D(x, a-D(G(z), a) + λ_GP \cdot GP+0.001 \cdot D(x,a)^2$. 

The loss for Generator is $loss_G = -D(G(z),a)$

## 9. SRGAN
<p align="center">
  <img src="https://i.ibb.co/YDZKWf0/7-SRGAN.png" width="300" height="200">
</p>

### 9.1 Abstract (2017)
Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains
largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this paper, we present SRGAN, a generative adversarial network (GAN) for image superresolution (SR). To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4x upscaling factors. To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks. An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN. The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art method.

### 9.2 Models: Generator and Discriminator

The architecture of Generator and Discriminator networks is as follows:

<p align="center">
  <img src="https://i.ibb.co/fFs7KkS/Picture1.png" width="800" height="500">
</p>

<p align="center">
    <em> Table 13: Discriminator of SRGAN </em>
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
    <em> Table 14: Generator of SRGAN </em>
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

### 9.3 Loss
The objective of the Discriminator and Generator are:
* Discriminator: $max [ log(D(x)) + log( 1-D(G(z)) ) ]$
* Generator: $min [ log(1-D(G(z)) ]$ or $max[ log(D(G(z)) ]$ 

The loss of the Discriminator and Generator are:
* Discriminator: $loss_D = BCE[ D(res_h), 1-0.01 \cdot D(res_h) ] + BCE[ D(G(res_l)), 0 ]$
* Generator: $loss_G = loss_{adversarial} + loss_{VGG} = 0.001 \cdot BCE[ D(G(res_l)), 1 )]+ 0.006 \cdot VGG[ G(res_l), res_h ]$

where $res_l$ is the low resolution image of size $3×24×24$ and $res_h$ is the high resolution image of size $3×96×96$.

## 10. ESRGAN
<p align="center">
  <img src="https://i.ibb.co/YDZKWf0/7-SRGAN.png" width="300" height="200">
</p>

### 10.1 Abstract (2018)
Abstract. The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating realistic textures during single image super-resolution. However, the hallucinated details are often accompanied with unpleasant artifacts. To further enhance the visual quality, we thoroughly study three key components of SRGAN network architecture, adversarial loss and perceptual loss, and improve each of them to derive an Enhanced SRGAN (ESRGAN). In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency and texture recovery. Beneting from these improvements, the proposed ESRGAN achieves consistently better visual quality with more realistic and natural textures than SRGAN and won the first place in the PIRM2018-SR Challenge.

### 10.2 Models: Generator and Discriminator

The Discriminator of ESRGAN is exactly the same as SRGAN.

The architecture of ESRGAN's Generator has two modifications: 
* Remove all Batch Normalization layers from SRGAN
* Replace original block with the proposed Residual-in-Residual Dense Block (RRDB)

<p align="center">
  <img src="https://i.ibb.co/72ZgvNF/Picture2.png" width="800" height="380">
</p>

<p align="center">
    <em> Table 15: Generator of ESRGAN </em>
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

### 10.3 Loss

The loss for Discriminator is $loss_D = -E[ D(res_h) ] - E[ D(G(res_l)) ] + λ_GP \cdot GP$

The loss for Generator is $loss_G=loss_{L1}+loss_{VGG}+loss_{adversarial}$ where
* $loss_{L1} = 0.01 \cdot L1 = 0.01 \cdot L1[G(res_l), res_h]$
* $loss_{VGG} = VGG[ G(res_l), res_h)  ] = MSE[ (G(res_l), res_h ]$ of VGG-19
* $loss_{adversarial} = -0.005E[ D(G(res_l)) ]$
