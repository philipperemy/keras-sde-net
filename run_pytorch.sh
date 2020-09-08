#!/bin/bash

set -e

cd MNIST
python -u resnet_mnist.py
python -u test_detection.py --pre_trained_net save_resnet_mnist/final_model --network resnet --dataset mnist --out_dataset svhn

python -u resnet_dropout_mnist.py
python -u test_detection.py --pre_trained_net save_resnet_dropout_mnist/final_model --network mc_dropout --dataset mnist --out_dataset svhn

python -u sdenet_mnist.py
python -u test_detection.py --pre_trained_net save_sdenet_mnist/final_model --network sdenet --dataset mnist --out_dataset svhn

cd ..
cd SVHN

python -u resnet_svhn.py
python -u test_detection.py --pre_trained_net save_resnet_svhn/final_model --network resnet --dataset svhn --out_dataset cifar10
python -u resnet_dropout_svhn.py
python -u test_detection.py --pre_trained_net save_resnet_dropout_svhn/final_model --network mc_dropout --dataset svhn --out_dataset cifar10

python -u sdenet_svhn.py
python -u test_detection.py --pre_trained_net save_sdenet_svhn_0/final_model --network sdenet --dataset svhn --out_dataset cifar10

cd ..
cd YearMSD

mkdir -p save_sdenet_msd
mkdir -p save_mc_msd

python -u DNN_mc.py
python -u test_detection_mc.py --pre_trained_net save_mc_msd/final_model
python -u SDE_regression.py
python -u test_detection_sde.py --pre_trained_net save_sdenet_msd/final_model
