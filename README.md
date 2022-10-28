# PSDA
## Probabilistic _Spherical_ Discriminant Analysis

**Update**: We have a new, more powerful generalization, **T-PSDA**. The new code repo is [here](https://github.com/bsxfan/Toroidal-PSDA), and a paper describing the new model is available [here](https://arxiv.org/abs/2210.15441).

This is a Python implementation of the algorithms described in our (submitted) Interspeech 2022 paper:
> [Probabilistic Spherical Discriminant Analysis: An Alternative to PLDA for length-normalized embeddings](https://arxiv.org/abs/2203.14893)

- Please cite this paper if you find our code useful.

Probabilistic _Linear_ Discrimnant Analysys (PLDA) is a trainable scoring backend that can be used for things like speaker/face recognition or clustering, or speaker diarization. PLDA uses the self-conjugacy of multivariate Gaussians to obtain closed-form scoring and closed-form EM updates for learning. Some of the Gaussian assumptions of the PLDA model are violated when embeddings are length-normalized.

With PSDA, we use [Von Mises-Fisher](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution) (VMF) instead of Gaussians, because they may give a better model for this kind of data. The VMF is also self-conjugate, so we enjoy the same benefits of closed-form scoring and EM-learning.

## Installation
For now everything is implemented in numpy and scipy. (The EM algorithm has closed-form updates, so we don't need automatic derivatives for now). The demo code uses our [PYLLR](https://github.com/bsxfan/PYLLR) toolkit for evaluation of the accuracy and calibration.

We will neaten the installation procedure later. For now, install PYLLR and then just put the directory of this toolkit in your python path. Then run demo.py to see that it works and look at the demo code to figure out how to use the toolkit for training and scoring.
