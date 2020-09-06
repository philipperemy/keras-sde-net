#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:34:10 2019

@author: lingkaikong
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.utils import to_categorical

from sdenet.data import data_loader
from sdenet.models.helpers import save_weights, set_seed, add_l2_weight_decay
from sdenet.models.sdenet_mnist import SDENet_mnist


def main():
    parser = argparse.ArgumentParser(description='PyTorch SDE-Net Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate of drift net')
    parser.add_argument('--lr2', default=0.01, type=float, help='learning rate of diffusion net')
    parser.add_argument('--training_out', action='store_false', default=True, help='training_with_out')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
    parser.add_argument('--eva_iter', default=5, type=int, help='number of passes when evaluation')
    parser.add_argument('--dataset_inDomain', default='mnist', help='training dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--seed', type=float, default=0)
    parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
    parser.add_argument('--decreasing_lr', default=[10, 20, 30], nargs='+', help='decreasing strategy')
    parser.add_argument('--decreasing_lr2', default=[15, 30], nargs='+', help='decreasing strategy')
    args = parser.parse_args()

    set_seed(args.seed)

    print('load in-domain data: ', args.dataset_inDomain)
    train_loader_inDomain, test_loader_inDomain = data_loader.getDataSet(args.dataset_inDomain, args.batch_size,
                                                                         args.test_batch_size, args.imageSize)

    # Model
    print('==> Building model..')
    net = SDENet_mnist(layer_depth=6, num_classes=10, dim=64)
    add_l2_weight_decay(net, weights_decay=5e-4)

    real_label = 0
    fake_label = 1

    criterion = MeanSquaredError()
    criterion2 = MeanSquaredError()

    optimizer_f = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)
    optimizer_g = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_loss_in = tf.keras.metrics.Mean(name='train_loss_in')
    train_loss_out = tf.keras.metrics.Mean(name='train_loss_out')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_loss_out')

    # use a smaller sigma during training for training stability
    # TODO: for now let's only focus on the non diffusion part.
    net.sigma = 20

    # Training
    def train(ep):

        print('\nEpoch: %d' % ep)
        # training with in-domain data
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader_inDomain):
            inputs, targets = np.array(inputs), np.array(targets)
            inputs = np.transpose(inputs, (0, 2, 3, 1))
            total += len(inputs)

            # optimizer F
            with tf.GradientTape() as tape:
                outputs = net(inputs)
                num_classes = outputs.shape[-1]
                outputs_softmax = K.softmax(outputs)
                loss = criterion(outputs_softmax, to_categorical(targets, num_classes))
                train_accuracy(targets, outputs_softmax)
                train_loss(loss)

                f_variables = net.downsampling_layers.trainable_variables
                f_variables += net.drift.trainable_variables
                f_variables += net.fc_layers.trainable_variables
                gradients = tape.gradient(loss, f_variables)
                optimizer_f.apply_gradients(zip(gradients, f_variables))

            # optimizer G
            with tf.GradientTape() as tape:
                g_variables = net.diffusion.trainable_variables
                label = K.ones(shape=(args.batch_size, 1)) * real_label  # real = 0.
                assert np.mean(label.numpy()) == real_label
                loss_in = criterion2(net(inputs, training_diffusion=True), label)
                train_loss_in(loss_in)
                gradients_in = tape.gradient(loss_in, g_variables)
                optimizer_g.apply_gradients(zip(gradients_in, g_variables))

            with tf.GradientTape() as tape:
                label = label * 0 + fake_label  # fake = 1.
                assert np.mean(label.numpy()) == fake_label
                inputs_out = 2 * K.random_normal((args.batch_size, args.imageSize, args.imageSize, 1)) + inputs
                loss_out = criterion2(net(inputs_out, training_diffusion=True), label)
                train_loss_out(loss_out)
                gradients_out = tape.gradient(loss_out, g_variables)
                optimizer_g.apply_gradients(zip(gradients_out, g_variables))

            # gradients_sum = [g1 + g2 for g1, g2 in zip(gradients_in, gradients_out)]
            # assert len(gradients_sum) == len(g_variables)

        print('Train epoch:{} \tLoss: {:.6f} | Loss_in: {:.8f}, Loss_out: {:.8f} | Acc: {:.4f}'
              .format(ep, train_loss.result(), train_loss_in.result(), train_loss_out.result(),
                      train_accuracy.result()))

    def test(ep):
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader_inDomain):
            inputs, targets = np.array(inputs), np.array(targets)
            inputs = np.transpose(inputs, (0, 2, 3, 1))
            outputs = 0
            for j in range(args.eva_iter):
                current_batch = net(inputs)
                outputs = outputs + K.softmax(current_batch, axis=1)

            outputs = outputs / args.eva_iter
            predicted = K.argmax(outputs)
            total += len(targets)
            correct += np.array(predicted == targets).sum()

        print('Test epoch: {} | Acc: {:.6f}'.format(ep, 100. * correct / total))

    for epoch in range(0, args.epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        train_loss_in.reset_states()
        train_loss_out.reset_states()

        train(epoch)
        test(epoch)

        if epoch in args.decreasing_lr:
            current_lr = float(optimizer_f._decayed_lr(tf.float32))
            new_lr = current_lr * args.droprate
            print(f'Current LR_F: {current_lr:.6f}, New LR_F: {new_lr:.6f}.')
            optimizer_f.lr.assign(current_lr * new_lr)

        if epoch in args.decreasing_lr2:
            current_lr = float(optimizer_g._decayed_lr(tf.float32))
            new_lr = current_lr * args.droprate
            print(f'Current LR_G: {current_lr:.6f}, New LR_G: {new_lr:.6f}.')
            optimizer_g.lr.assign(current_lr * new_lr)

        output_dir = Path('save_sdenet_mnist')
        output_dir.mkdir(parents=True, exist_ok=True)
        save_weights(net, str(output_dir / 'final_model.h5'))


if __name__ == '__main__':
    main()
