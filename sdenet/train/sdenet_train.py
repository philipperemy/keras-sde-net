#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:34:10 2019

@author: philipperemy
"""
import argparse

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy

from sdenet.data import data_loader
from sdenet.models.helpers import set_seed, add_l2_weight_decay, Checkpoints
from sdenet.models.sdenet import SDENet


def main():
    parser = argparse.ArgumentParser(description='Keras SDE-Net Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate of drift net')
    parser.add_argument('--lr2', default=None, type=float, help='learning rate of diffusion net')
    parser.add_argument('--training_out', action='store_false', default=True, help='training_with_out')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs to train')
    parser.add_argument('--eva_iter', default=5, type=int, help='number of passes when evaluation')
    parser.add_argument('--dataset', default=None, help='training dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--imageSize', type=int, default=None, help='the height / width of the input image to network')
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--seed', type=float, default=0)
    parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
    parser.add_argument('--decreasing_lr', default=None, nargs='+', help='decreasing strategy')
    parser.add_argument('--decreasing_lr2', default=None, nargs='+', help='decreasing strategy')
    parser.add_argument('--net_sigma', default=50, type=int, help='sigma coefficient for the diffusion')
    parser.add_argument('--net_save_dir', default='net', type=str, help='where to save the checkpoints')
    args = parser.parse_args()
    print(args)

    if args.seed != 0:
        set_seed(args.seed)

    print('load in-domain data: ', args.dataset)
    apply_grayscale = args.dataset == 'mnist'
    train_loader_in_domain, test_loader_in_domain = data_loader.getDataSet(args.dataset, args.batch_size,
                                                                           args.test_batch_size, args.imageSize,
                                                                           apply_grayscale=apply_grayscale)

    # Model
    print('==> Building model..')
    net = SDENet(layer_depth=6, num_classes=10, dim=64, task=args.dataset)
    add_l2_weight_decay(net, weights_decay=5e-4)

    real_label = 0
    fake_label = 1

    criterion = SparseCategoricalCrossentropy(from_logits=True)
    criterion2 = BinaryCrossentropy()  # from_logits=False.

    optimizer_f = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)
    optimizer_g = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_loss_in = tf.keras.metrics.Mean(name='train_loss_in')
    train_loss_out = tf.keras.metrics.Mean(name='train_loss_out')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # use a smaller sigma during training for training stability
    net.set_sigma(sigma=args.net_sigma)

    # Training
    def train(ep):
        ones = K.ones(shape=(args.batch_size, 1))

        # tf.function -> faster execution but cant debug.
        @tf.function
        def train_step(inp, tar):
            # optimizer F
            with tf.GradientTape() as tape:
                outputs = net(inp)
                loss = criterion(tar, outputs)
                train_accuracy(tar, outputs)
                train_loss(loss)

                f_variables = net.downsampling_layers.trainable_variables
                f_variables += net.drift.trainable_variables
                f_variables += net.fc_layers.trainable_variables
                gradients = tape.gradient(loss, f_variables)
                optimizer_f.apply_gradients(zip(gradients, f_variables))

            # optimizer G
            # training with out-of-domain data
            if args.training_out:
                with tf.GradientTape() as tape:
                    g_variables = net.diffusion.trainable_variables
                    label = ones * real_label  # real = 0.
                    loss_in = criterion2(label, net(inp, training_diffusion=True))
                    train_loss_in(loss_in)
                    gradients_in = tape.gradient(loss_in, g_variables)
                    optimizer_g.apply_gradients(zip(gradients_in, g_variables))

                with tf.GradientTape() as tape:
                    label = label * 0 + fake_label  # fake = 1.
                    inputs_out = 2 * K.random_normal((args.batch_size, args.imageSize, args.imageSize, 1)) + inp
                    loss_out = criterion2(label, net(inputs_out, training_diffusion=True))
                    train_loss_out(loss_out)
                    gradients_out = tape.gradient(loss_out, g_variables)
                    optimizer_g.apply_gradients(zip(gradients_out, g_variables))

        print('\nEpoch: %d' % ep)
        # training with in-domain data
        for batch_idx, (inputs, targets) in enumerate(train_loader_in_domain):
            inputs, targets = np.array(inputs), np.array(targets)
            inputs = np.transpose(inputs, (0, 2, 3, 1))
            train_step(inputs, targets)

        print('Train epoch:{} \tLoss: {:.6f} | Loss_in: {:.8f}, Loss_out: {:.8f} | Acc: {:.4f}'
              .format(ep, train_loss.result(), train_loss_in.result(), train_loss_out.result(),
                      train_accuracy.result()))

    def test(ep):

        @tf.function
        def test_step(inp, tar):
            outputs = 0
            for j in range(args.eva_iter):
                current_batch = net(inp)
                outputs = outputs + K.softmax(current_batch, axis=1)
            outputs = outputs / args.eva_iter
            test_accuracy(tar, outputs)

        for batch_idx, (inputs, targets) in enumerate(test_loader_in_domain):
            inputs, targets = np.array(inputs), np.array(targets)
            inputs = np.transpose(inputs, (0, 2, 3, 1))
            test_step(inputs, targets)

        print('Test epoch: {} | Acc: {:.6f}'.format(ep, test_accuracy.result()))

    net_save_dir = f'{args.net_save_dir}_{args.seed}' if args.seed != 0 else args.net_save_dir
    checkpoints = Checkpoints(net, net_save_dir)
    for epoch in range(0, args.epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        train_loss_in.reset_states()
        train_loss_out.reset_states()
        test_accuracy.reset_states()

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

        channels = 3 if args.dataset != 'mnist' else 1
        checkpoints.persist(float(test_accuracy.result()), [1, args.imageSize, args.imageSize, channels])


if __name__ == '__main__':
    main()
