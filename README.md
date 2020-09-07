# SDE Net (Keras)
This repo contains the code for the paper:

Lingkai Kong, Jimeng Sun and Chao Zhang, SDE-Net: Equipping Deep Neural Network with Uncertainty Estimates, ICML2020.

[[paper](https://arxiv.org/abs/2008.10546)] [[video](https://www.youtube.com/watch?v=RylZA4Ioc3M)]

![SDE-Net](figure/illustration.png)

## Package installation

```bash
virtualenv -p python3 venv && source venv/bin/activate # optional but recommended.
pip install -r requirements.txt && pip install -e . # install the package.
```

## Training & Evaluation

#### MNIST

Training vanilla ResNet:

```bash
python sdenet/train/resnet_train.py 
```

Evaluation:

```bash
python sdenet/eval/eval_detection.py --pre_trained_net save_resnet_mnist/final_model.h5 --network resnet --dataset mnist --out_dataset svhn
```

Training SDE-Net:

```bash
python sdenet/train/sdenet_train.py 
```

Evaluation:

```bash
python sdenet/eval/eval_detection.py --pre_trained_net save_sdenet_mnist/final_model.h5 --network sdenet --dataset mnist --out_dataset svhn
```
