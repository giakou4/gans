# Generative Adversarial Networks (GANs)

Implementation of GAN architectures in [PyTorch](https://pytorch.org/)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-PyTorch-red)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/gans/LICENSE)

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

<div align="center">

|   |   | **Layer** | **Activation** | **Feature Map** | **Size** |
|---|---|:---------:|:--------------:|:---------------:|:--------:|
|   |   | INPUT (FLATTEN IMAGE)     |                |                 |   1×784  |
| 1 |   | LINEAR    |   Leaky ReLU   |       128       |   1×128  |
| 2 |   | LINEAR    |     Sigmoid    |        1        |    1×1   |
|   |   | OUTPUT (RPOBABILITY)    |                |                 |     1    |

</div>

<p align="center">
    <em> Table 2: Generator of a Simple GAN </em>
</p>

<div align="center">

|   |   | **Layer** | **Activation** | **Feature Map** | **Size** |
|---|---|:---------:|:--------------:|:---------------:|:--------:|
|   |   | INPUT (NOISE)    |                |                 |   1×64   |
| 1 |   | LINEAR    |   Leaky ReLU   |       256       |   1×256  |
| 2 |   | LINEAR    |      Tanh      |       784       |   1×784  |
|   |   | OUTPUT (FLATTEN IMAGE)    |                |                 |   1×784  |

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
    <em> Table 5: Critic of WCGAN </em>
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
    <em> Table 6: Generator of WCGAN </em>
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

## 4. WGAN-GP (with Gradient Penalty)
<p align="center">
  <img src="https://i.ibb.co/BBtZ2Wc/2-WGAN.png" width="300" height="200">
</p>

A new paper introduced the WGAN with Gradient Penalty (GP), where in the loss function of Critic, a penalty is added. WGAN and WGAN-GP use the exact same models for Generator and Critic. The only difference lies is the loss back propagated.

### 4.1 Abstract (2017)
Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The recently proposedWasserstein GAN (WGAN) makes progress toward stable training of GANs, but sometimes can still generate only poor samples or fail to converge. We find that these problems are often due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior. We propose an alternative to clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no hyperparameter tuning, including 101-layer ResNets and language models with continuous generators. We also achieve high quality generations on CIFAR-10 and LSUN bedrooms.

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
<p align="center">
  <img src="https://i.ibb.co/ChcTqGc/4-UNET.jpg" width="535" height="335">
</p>

### 6.3 Loss


## 7. CycleGAN
<p align="center">
  <img src="https://i.ibb.co/mHpVJDh/5-Cycle-GAN.png"  width="500" height="350">
</p>

### 7.1 Abstract (2020)
Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G: X->Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F:Y->X and introduce a cycle consistency loss to enforce F(G(X))->X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.

### 7.2 Models: Generator and Discriminator

### 7.3 Loss

## 8. ProGAN 
<p align="center">
  <img src="https://i.ibb.co/2vZp68R/6-ProGAN.png" width="200" height="350">
</p>

### 8.1 Abstract (2018)
We describe a new training methodology for generative adversarial networks. The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality, e.g., CELEBA images at 10242. We also propose a simple way to increase the variation in generated images, and achieve a record inception score of 8:80 in unsupervised CIFAR10. Additionally, we describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator. Finally, we suggest a new metric for evaluating GAN results, both in terms of image quality and variation. As an additional contribution, we construct a higher-quality version of the CELEBA dataset.

### 8.2 Models: Generator and Discriminator

### 8.3 Loss

## 9. SRGAN
<p align="center">
  <img src="https://i.ibb.co/YDZKWf0/7-SRGAN.png" width="300" height="200">
</p>

### 9.1 Abstract (2017)
Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains
largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this paper, we present SRGAN, a generative adversarial network (GAN) for image superresolution (SR). To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4x upscaling factors. To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks. An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN. The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art method.

### 9.2 Models: Generator and Discriminator

### 9.3 Loss

## 10. ESRGAN
<p align="center">
  <img src="https://i.ibb.co/YDZKWf0/7-SRGAN.png" width="300" height="200">
</p>

### 10.1 Abstract (2018)
Abstract. The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating realistic textures during single image super-resolution. However, the hallucinated details are often accompanied with unpleasant artifacts. To further enhance the visual quality, we thoroughly study three key components of SRGAN network architecture, adversarial loss and perceptual loss, and improve each of them to derive an Enhanced SRGAN (ESRGAN). In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency and texture recovery. Beneting from these improvements, the proposed ESRGAN achieves consistently better visual quality with more realistic and natural textures than SRGAN and won the first place in the PIRM2018-SR Challenge.

### 10.2 Models: Generator and Discriminator

### 10.3 Loss
