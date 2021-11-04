# Self Supervised Vision
**This repo will always be a work in progress!**

## About
<a name="about"></a>
Implementations and observations on some self-supervised methods used to pre-train networks for vision tasks. This is an exploratory project; these experiments were run to develop some intuition about the working of and differences between these methods. The shown performance scores might not be SOTA, especially considering most authors have trained on the ILSVRC 2012 (ImageNet) dataset, whereas I have trained on CIFAR-10.

## Algorithms
<a name="algorithms"></a>
The implementations of algorithms in the table below are currently in progress (or have been completed, if the KNN accuracy is mentioned). This is not an exhaustive list: I'll add more papers to it as I find them.

|                                   Algorithm                                   |  Shorthand  |                     Paper                     | KNN accuracy |
|:-----------------------------------------------------------------------------:|:-----------:|:---------------------------------------------:|:------------:|
| Bootstrap Your Own Latent: A new approach to self-supervised Learning         |     BYOL    |   [arXiv](https://arxiv.org/abs/2006.07733)   |    80.09     | 
| Representation Learning via Invariant Causal Mechanisms                       |    ReLIC    |   [arXiv](https://arxiv.org/abs/2010.07922)   |    79.26     |
| A Simple Framework for Contrastive Learning of Visual Representations         |    SimCLR   |   [arXiv](https://arxiv.org/abs/2002.05709)   |    77.79     |
| Unsupervised Learning of Visual Features by Contrasting Cluster Assignments   |     SwAV    |   [arXiv](https://arxiv.org/abs/2006.09882)   |    72.11     |
| Momentum Contrast for Unsupervised Visual Representation Learning             |     MoCo    |   [arXiv](https://arxiv.org/abs/1911.05722)   |    63.14     |
| Barlow Twins: Self-Supervised Learning via Redundancy Reduction               |    Barlow   |   [arXiv](https://arxiv.org/abs/2103.03230)   |    56.81     |
| Self-Supervised Learning of Pretext-Invariant Representations                 |     PIRL    |   [arXiv](https://arxiv.org/abs/1912.01991)   |              |
| Learning Representations by Maximizing Mutual Information across Views        |    AMDIM    |   [arXiv](https://arxiv.org/abs/1906.00910)   |              |
| Representation Learning with Contrastive Predictive Coding                    |     CPC     |   [arXiv](https://arxiv.org/abs/1807.03748)   |              |  
| Self-Supervised Pretraining of Visual Features in the Wild                    |     SEER    |   [arXiv](https://arxiv.org/abs/2103.01988)   |              |
| Self-labelling via simultaneous clustering and representation learning        |     SeLa    |   [arXiv](https://arxiv.org/abs/1911.05371)   |              |
| Emerging Properties in Self-Supervised Vision Transformers                    |     DINO    |   [arXiv](https://arxiv.org/abs/2104.14294)   |              |

## Usage
<a name="usage"></a>
To train any of these models on your end (on CIFAR10), run this command inside `Self-Supervised-Vision/`.

```
python main.py --config configs/<model>.yaml --algo <model> --arch <arch> --task train
```

Here, `<model>` is the algorithm's shorthand in lowercase (refer [Algorithms](#markdown-header-algorithms)) and `<arch>` is one of `resnet18, resnet50, resnext50, resnext101, wide_resnet50, wide_resnet101` (`resnet18` by default).