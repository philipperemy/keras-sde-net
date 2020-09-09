#!/bin/bash

set -e

# RESNET - MNIST
python -u sdenet/train/resnet_train.py --epochs 40 --dataset "mnist" --imageSize 28 --decreasing_lr 10 20 30 --net_save_dir "save_resnet_mnist"
python -u sdenet/eval/eval_detection.py --pre_trained_net "save_resnet_mnist/best_model.h5" --network "resnet" --dataset "mnist" --out_dataset "svhn" --imageSize 28

# SDENET - MNIST
python -u sdenet/train/sdenet_train.py --epochs 40 --dataset "mnist" --imageSize 28 --decreasing_lr 10 20 30 --lr2 0.01 --decreasing_lr2 15 30 --net_sigma 20 --net_save_dir "save_sdenet_mnist"
python -u sdenet/eval/eval_detection.py --pre_trained_net "save_sdenet_mnist/best_model.h5" --network "sdenet" --dataset "mnist" --out_dataset "svhn" --imageSize 28

# RESNET - SVHN
python -u sdenet/train/resnet_train.py --epochs 60 --dataset "svhn"  --imageSize 32 --decreasing_lr 20 40 --net_save_dir "save_resnet_svhn"
python -u sdenet/eval/eval_detection.py --pre_trained_net "save_resnet_svhn/best_model.h5" --network "resnet" --dataset "svhn" --out_dataset "cifar10" --imageSize 32

# SDENET - SVHN
python -u sdenet/train/sdenet_train.py --epochs 60 --dataset "svhn"  --imageSize 32 --decreasing_lr 20 40 --lr2 0.005 --decreasing_lr2 10 30 --net_sigma 5 --net_save_dir "save_sdenet_svhn"
python -u sdenet/eval/eval_detection.py --pre_trained_net "save_sdenet_svhn/best_model.h5" --network "sdenet" --dataset "svhn" --out_dataset "cifar10" --imageSize 32
