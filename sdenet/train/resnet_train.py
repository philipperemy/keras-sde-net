import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from sdenet.data import data_loader
from sdenet.models.helpers import set_seed, Checkpoints
from sdenet.models.resnet import ResidualNet


class MNISTProfile:
    epochs = 40
    dataset = 'mnist'
    imageSize = 28
    decreasing_lr = [10, 20, 30]
    net_save_dir = 'save_resnet_mnist'


class SVHNProfile:
    epochs = 60
    dataset = 'svhn'
    imageSize = 32
    decreasing_lr = [20, 40]
    net_save_dir = 'save_resnet_svhn'


def apply_profile_to_args(args, profile):
    if args.epochs is None:
        args.epochs = profile.epochs
    if args.dataset is None:
        args.dataset = profile.dataset
    if args.imageSize is None:
        args.imageSize = profile.imageSize
    if args.decreasing_lr is None:
        args.decreasing_lr = profile.decreasing_lr


def main():
    # Mostly from: https://www.tensorflow.org/tutorials/quickstart/advanced
    parser = argparse.ArgumentParser(description='Keras ResNet Training')
    parser.add_argument('--task', required=True, choices=['mnist', 'svhn'])
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--dataset', default='mnist', help='cifar10 | svhn')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
    parser.add_argument('--decreasing_lr', default=[10, 20, 30], nargs='+', help='decreasing strategy')
    parser.add_argument('--seed', type=float, default=0)
    args = parser.parse_args()

    # Apply profile.
    if args.task == 'mnist':
        profile = MNISTProfile
    elif args.task == 'svhn':
        profile = SVHNProfile
    else:
        raise Exception(f'Unknown task: {args.task}.')
    apply_profile_to_args(args, profile)
    print(args)

    if args.seed != 0:
        set_seed(args.seed)

    print(f'load data: {args.dataset}')
    train_loader, test_loader = data_loader.getDataSet(args.dataset, args.batch_size,
                                                       args.test_batch_size, args.imageSize)

    # Model
    print('==> Building model..')
    net = ResidualNet()

    criterion = SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = net(images, training=True)
            loss = criterion(labels, predictions)
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = net(images, training=False)
        t_loss = criterion(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    net_save_dir = f'{profile.net_save_dir}_{args.seed}' if args.seed != 0 else profile.net_save_dir
    checkpoints = Checkpoints(net, net_save_dir)
    for epoch in range(0, args.epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        template = 'Epoch {}, Loss: {:.3f}, Accuracy: {:.3f}, Test Loss: {:.3f}, Test Accuracy: {:.3f}'
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.numpy(), targets.numpy()
            inputs = np.transpose(inputs, (0, 2, 3, 1))
            train_step(inputs, targets)
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.numpy(), targets.numpy()
            inputs = np.transpose(inputs, (0, 2, 3, 1))
            test_step(inputs, targets)
        print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100,
                              test_loss.result(), test_accuracy.result() * 100))

        if epoch in args.decreasing_lr:
            current_lr = float(optimizer._decayed_lr(tf.float32))
            new_lr = current_lr * args.droprate
            print(f'Current LR: {current_lr:.6f}, New LR: {new_lr:.6f}.')
            optimizer.lr.assign(current_lr * new_lr)

        checkpoints.persist(float(test_accuracy.result()))


if __name__ == '__main__':
    main()
