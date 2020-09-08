#!/bin/bash

set -e

# RESNET - MNIST
python -u sdenet/train/resnet_train.py --task mnist
python -u sdenet/eval/eval_detection.py --pre_trained_net save_resnet_mnist/best_model.h5 --network resnet --dataset mnist --out_dataset svhn

# SDENET - MNIST
python -u sdenet/train/sdenet_train.py --task mnist
python -u sdenet/eval/eval_detection.py --pre_trained_net save_sdenet_mnist/best_model.h5 --network sdenet --dataset mnist --out_dataset svhn

# RESNET - SVHN
python -u sdenet/train/resnet_train.py --task svhn
python -u sdenet/eval/eval_detection.py --pre_trained_net save_resnet_svhn/best_model.h5 --network resnet --dataset svhn --out_dataset cifar10

# SDENET - SVHN
python -u sdenet/train/sdenet_train.py --task svhn
python -u sdenet/eval/eval_detection.py --pre_trained_net save_sdenet_svhn/best_model.h5 --network sdenet --dataset svhn --out_dataset cifar10
