## SDE Net (Keras)
This repo contains the code for the paper:

Lingkai Kong, Jimeng Sun and Chao Zhang, SDE-Net: Equipping Deep Neural Network with Uncertainty Estimates, ICML2020.

[[paper](https://arxiv.org/abs/2008.10546)] [[video](https://www.youtube.com/watch?v=RylZA4Ioc3M)]

![SDE-Net](figure/illustration.png)

### Package installation

From PyPI

```bash
pip install sdenet
```

From the sources

```bash
git clone https://github.com/philipperemy/keras-sde-net.git && cd keras-sde-net
virtualenv -p python3 venv && source venv/bin/activate # optional but recommended.
pip install -r requirements.txt && pip install -e . # install the package.
```

### Import the models

```python
from sdenet import SDENet
from sdenet import ResidualNet
```

### Training & Evaluation

Supported datasets are: MNIST, SVHN, CIFAR10, CIFAR100. Supported models are RESNET and SDENET.

Look at the bash run scripts at the root of the repository to get started for training and evaluation.


### Comparison between official Pytorch implementation and Keras

This comparison is just the result of one run. No runs were handpicked. Overall it's very similar.

Except probably SDENET on SVHN (95% vs 94%).

#### Pytorch

```
MNIST RESNET
_________________________________

Final Accuracy: 9945/10000 (99.45%)

generate log  from out-of-distribution data
calculate metrics for OOD
OOD  Performance of Baseline detector
TNR at TPR 95%:            88.783%
AUROC:                     95.939%
Detection acc:             92.169%
AUPR In:                   86.441%
AUPR Out:                  98.434%

calculate metrics for mis
mis  Performance of Baseline detector
TNR at TPR 95%:            89.791%
AUROC:                     97.510%
Detection acc:             93.041%
AUPR In:                   99.985%
AUPR Out:                  34.000%


MNIST SDENET
_________________________________

Final Accuracy: 9927/10000 (99.27%)

generate log  from out-of-distribution data
calculate metrics for OOD
OOD  Performance of Baseline detector
TNR at TPR 95%:            99.372%
AUROC:                     99.804%
Detection acc:             98.692%
AUPR In:                   99.483%
AUPR Out:                  99.887%
calculate metrics for mis
mis  Performance of Baseline detector
TNR at TPR 95%:            92.544%
AUROC:                     97.525%
Detection acc:             94.485%
AUPR In:                   99.979%
AUPR Out:                  41.739%


SVHN RESNET
_________________________________

Final Accuracy: 24609/25856 (95.18%)

generate log  from out-of-distribution data
calculate metrics for OOD
OOD  Performance of Baseline detector
TNR at TPR 95%:            66.552%
AUROC:                     94.421%
Detection acc:             90.136%
AUPR In:                   97.639%
AUPR Out:                  84.998%
calculate metrics for mis
mis  Performance of Baseline detector
TNR at TPR 95%:            64.376%
AUROC:                     90.458%
Detection acc:             85.371%
AUPR In:                   99.301%
AUPR Out:                  44.899%


SVHN SDENET
_________________________________

Final Accuracy: 24588/25856 (95.10%)

generate log  from out-of-distribution data
calculate metrics for OOD
OOD  Performance of Baseline detector
TNR at TPR 95%:            65.215%
AUROC:                     94.308%
Detection acc:             89.746%
AUPR In:                   97.694%
AUPR Out:                  84.017%
calculate metrics for mis
mis  Performance of Baseline detector
TNR at TPR 95%:            67.831%
AUROC:                     91.267%
Detection acc:             86.501%
AUPR In:                   99.270%
AUPR Out:                  48.871%

```

#### Keras
```
MNIST RESNET
_________________________________

 Final Accuracy: 9944/10000 (99.44%)

generate log  from out-of-distribution data
calculate metrics for OOD
OOD  Performance of Baseline detector
TNR at TPR 95%:            93.162%
AUROC:                     97.946%
Detection acc:             94.250%
AUPR In:                   94.842%
AUPR Out:                  99.215%
calculate metrics for mis
mis  Performance of Baseline detector
TNR at TPR 95%:            96.997%
AUROC:                     98.863%
Detection acc:             96.697%
AUPR In:                   99.994%
AUPR Out:                  26.744%

MNIST SDENET
_________________________________

Final Accuracy: 9934/10000 (99.34%)

generate log  from out-of-distribution data
calculate metrics for OOD
OOD  Performance of Baseline detector
TNR at TPR 95%:            98.425%
AUROC:                     99.567%
Detection acc:             97.804%
AUPR In:                   98.613%
AUPR Out:                  99.872%
calculate metrics for mis
mis  Performance of Baseline detector
TNR at TPR 95%:            95.515%
AUROC:                     98.763%
Detection acc:             95.825%
AUPR In:                   99.992%
AUPR Out:                  32.524%

SVHN RESNET
_________________________________

 Final Accuracy: 24487/25856 (94.71%)

generate log  from out-of-distribution data
calculate metrics for OOD
OOD  Performance of Baseline detector
TNR at TPR 95%:            56.648%
AUROC:                     93.602%
Detection acc:             87.504%
AUPR In:                   97.627%
AUPR Out:                  81.664%
calculate metrics for mis
mis  Performance of Baseline detector
TNR at TPR 95%:            63.765%
AUROC:                     91.843%
Detection acc:             85.721%
AUPR In:                   99.386%
AUPR Out:                  46.231%

SVHN SDENET
_________________________________

 Final Accuracy: 24339/25856 (94.13%)

generate log  from out-of-distribution data
calculate metrics for OOD
OOD  Performance of Baseline detector
TNR at TPR 95%:            64.491%
AUROC:                     94.358%
Detection acc:             88.711%
AUPR In:                   97.776%
AUPR Out:                  87.517%
calculate metrics for mis
mis  Performance of Baseline detector
TNR at TPR 95%:            60.160%
AUROC:                     88.955%
Detection acc:             85.735%
AUPR In:                   99.165%
AUPR Out:                  45.268%
```
