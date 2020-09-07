#!/bin/bash

python -u sdenet/train/resnet_train.py --task mnist
python -u sdenet/eval/eval_detection.py --pre_trained_net save_resnet_mnist/final_model.h5 --network resnet --dataset mnist --out_dataset svhn

python -u sdenet/train/sdenet_train.py --task mnist
python -u sdenet/eval/eval_detection.py --pre_trained_net save_sdenet_mnist/final_model.h5 --network sdenet --dataset mnist --out_dataset svhn

python -u sdenet/train/resnet_train.py --task svhn
python -u sdenet/eval/eval_detection.py --pre_trained_net save_resnet_mnist/final_model.h5 --network resnet --dataset svhn --out_dataset cifar10

python -u sdenet/train/sdenet_train.py --task svhn
python -u sdenet/eval/eval_detection.py --pre_trained_net save_resnet_mnist/final_model.h5 --network sdenet --dataset svhn --out_dataset cifar10
