# Conditional GAN

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-PyTorch-red)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/gans/LICENSE)

<p align="center">
  <img src="https://i.ibb.co/JkHZqvY/3-Conditional-GAN.png" width="300" height="200">
</p>

## Abstract (2017)
Generative Adversarial Nets were recently introduced as a novel way to train generative models. In this work we introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data, y, we wish to condition on to both the generator and discriminator. We show that this model can generate MNIST digits conditioned on class labels. We also illustrate how this model could be used to learn a multi-modal model, and provide preliminary examples of an application to image tagging in which we demonstrate how this approach can generate descriptive tags which are not part of training labels.

## Generator and Critic
Conditional GANs have information about the label, so both Discriminator and Generator are supervised. The Generator generates what you want him to generate. It can be seen as an extension of WGAN. Discriminator includes an embedding from number of classes $num-classes$ to the image’s size squared $(img-size)^2$. The labels are embedded in that layer and resized to $img-size×img-size$. As a result, the first CONV 2D layer has 3+1 channels. Generator also includes an embedding. The labels are embedded and unsqueezed twice producing an extra channel. As a result, the first CONV 2D layer has 3+1 channels.