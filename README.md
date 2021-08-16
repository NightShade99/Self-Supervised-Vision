# Self Supervised Vision
**\[WIP\]** Implementations and observations on some self-supervised methods used to pre-train networks for vision tasks. This is meant to be a personal exploratory project; these implementations are not official.

## Algorithms
Tentatively, the algorithms checked in the table below will be implemented. Status key: `I`: Implemented but not run, `C`: Implemented and run, **Blank**: Yet to implement.

|                                   Algorithm                                   |  Shorthand  |                     Paper                     | Status | KNN accuracy | Linear eval |
|:-----------------------------------------------------------------------------:|:-----------:|:---------------------------------------------:|:------:|:------------:|:-----------:|
| Self-Supervised Learning of Pretext-Invariant Representations                 |     PIRL    |   [arXiv](https://arxiv.org/abs/1912.01991)   |        |              |             |
| Learning Representations by Maximizing Mutual Information across Views        |    AMDIM    |   [arXiv](https://arxiv.org/abs/1906.00910)   |        |              |             | 
| Representation Learning with Contrastive Predictive Coding                    |     CPC     |   [arXiv](https://arxiv.org/abs/1807.03748)   |        |              |             | 
| Deep Clustering for Unsupervised Learning of Visual Features                  | DeepCluster |   [arXiv](https://arxiv.org/abs/1807.05520)   |        |              |             | 
| ClusterFit: Improving Generalization of Visual Representations                |  ClusterFit |   [arXiv](https://arxiv.org/abs/1912.03330)   |        |              |             | 
| Unsupervised Learning of Visual Features by Contrasting Cluster Assignments   |     SwAV    |   [arXiv](https://arxiv.org/abs/2006.09882)   |        |              |             | 
| Self-Supervised Pretraining of Visual Features in the Wild                    |     SEER    |   [arXiv](https://arxiv.org/abs/2103.01988)   |        |              |             | 
| Unsupervised Feature Learning by Non-Parametric Instance Level Discrimination |   InstDisc  |   [arXiv](https://arxiv.org/abs/1805.01978)   |        |              |             | 
| Momentum Contrast for Unsupervised Visual Representation Learning             |     MoCo    |   [arXiv](https://arxiv.org/abs/1911.05722)   |  `C`   |    63.14     |             | 
| A Simple Framework for Contrastive Learning of Visual Representations         |    SimCLR   |   [arXiv](https://arxiv.org/abs/2002.05709)   |  `C`   |    77.79     |             | 
| Bootstrap Your Own Latent: A new approach to self-supervised Learning         |     BYOL    |   [arXiv](https://arxiv.org/abs/2006.07733)   |  `I`   |              |             | 
| Representation Learning via Invariant Causal Mechanisms                       |    ReLIC    |   [arXiv](https://arxiv.org/abs/2010.07922)   |        |              |             | 
| Exploring Simple Siamese Representation Learning                              |   SimSiam   |   [arXiv](https://arxiv.org/abs/2011.10566)   |        |              |             | 
| Emerging Properties in Self-Supervised Vision Transformers                    |     DINO    |   [arXiv](https://arxiv.org/abs/2104.14294)   |  `I`   |              |             | 
| Barlow Twins: Self-Supervised Learning via Redundancy Reduction               |    Barlow   |   [arXiv](https://arxiv.org/abs/2103.03230)   |        |              |             | 