# Self Supervised Vision

## About
<a name="about"></a>
Implementations and observations on some self-supervised methods used to pre-train networks for vision tasks. This is an exploratory project; these experiments were run to develop some intuition about the working of and differences between these methods. The shown performance scores might not be SOTA, especially considering most authors have trained on the ILSVRC 2012 (ImageNet) dataset, whereas I have trained on CIFAR-10.

## Algorithms
<a name="algorithms"></a>
The implementations of algorithms below are available. KNN accuracies are computed by clustering global features of images in CIFAR-10 test set and computing the average percentage of images in each cluster that belong to the same class (after Hungarian matching).

|                                   Algorithm                                   |  Shorthand  |                     Paper                     | KNN accuracy |
|:-----------------------------------------------------------------------------:|:-----------:|:---------------------------------------------:|:------------:|
| Bootstrap Your Own Latent: A new approach to self-supervised Learning         |     BYOL    |   [arXiv](https://arxiv.org/abs/2006.07733)   |    80.09     | 
| Representation Learning via Invariant Causal Mechanisms                       |    ReLIC    |   [arXiv](https://arxiv.org/abs/2010.07922)   |    79.26     |
| A Simple Framework for Contrastive Learning of Visual Representations         |    SimCLR   |   [arXiv](https://arxiv.org/abs/2002.05709)   |    77.79     |
| Unsupervised Learning of Visual Features by Contrasting Cluster Assignments   |     SwAV    |   [arXiv](https://arxiv.org/abs/2006.09882)   |    72.11     |
| Momentum Contrast for Unsupervised Visual Representation Learning             |     MoCo    |   [arXiv](https://arxiv.org/abs/1911.05722)   |    63.14     |
| Barlow Twins: Self-Supervised Learning via Redundancy Reduction               |    Barlow   |   [arXiv](https://arxiv.org/abs/2103.03230)   |    56.81     |

## Usage
<a name="usage"></a>
To train any of these models on your end (on CIFAR10), run this command inside `Self-Supervised-Vision/`.

```
python main.py --config configs/<model>.yaml --algo <model> --arch <arch> --task train
```

Here, `<model>` is the algorithm's shorthand in lowercase (refer [Algorithms](#markdown-header-algorithms)) and `<arch>` is one of `resnet18, resnet50, resnext50, resnext101, wide_resnet50, wide_resnet101` (`resnet18` by default).
