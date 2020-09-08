import pickle
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.models import Model
from tensorflow.python.keras.regularizers import l2


def fc(num_units, activation=None):
    return Dense(
        units=num_units,
        kernel_initializer=initializers.RandomNormal(stddev=1e-3),
        bias_initializer=initializers.Zeros(),
        activation=activation
    )


def conv3x3(out_planes, stride=1):
    # if padding=x, then add a ZeroPadding2D((x,x)) layer
    return Conv2D(filters=out_planes, strides=stride, kernel_size=3, padding='same', use_bias=False,
                  kernel_initializer='he_normal', bias_initializer='zeros')


def conv1x1(out_planes, stride=1):
    # if padding=x, then add a ZeroPadding2D((x,x)) layer
    return Conv2D(out_planes, kernel_size=1, strides=(stride, stride), use_bias=False,
                  kernel_initializer='he_normal', bias_initializer='zeros')


def norm(dim):
    # https://www.tensorflow.org/addons/tutorials/layers_normalizations
    num_groups = min(32, dim)
    return tfa.layers.GroupNormalization(groups=num_groups, axis=-1)


def set_seed(random_seed=123):
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)


def save_weights(d: Model, filename: str, input_shape: list):
    print(f'Saving weights to: {filename}.')
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    weights = d.get_weights()
    weights.append(input_shape)  # pop last = input shape (my convention).
    with open(filename, 'wb') as w:
        pickle.dump(weights, w)


def load_weights(d: Model, filename):
    assert Path(filename).exists()
    with open(filename, 'rb') as r:
        weights = list(pickle.load(r))
        input_shape = weights.pop()  # pop last = input shape (my convention).
        d(np.ones(shape=input_shape))  # forward pass to compute input shapes.
        d.set_weights(weights)


def add_l2_weight_decay(net: Model, weights_decay=5e-4):
    reg = l2(weights_decay)
    for layer in net.layers:
        if isinstance(layer, Model):
            add_l2_weight_decay(layer, weights_decay)
        for attr in ['kernel_regularizer', 'bias_regularizer']:
            if hasattr(layer, attr) and layer.trainable:
                print(f'Set l2 regularizer to layer {layer.name} : L2({weights_decay}).')
                setattr(layer, attr, reg)


class Checkpoints:

    def __init__(self, net, net_save_dir):
        self.net = net
        self.best_test_accuracy = 0.0
        self.output_dir = Path(net_save_dir)
        if self.output_dir.exists():
            shutil.rmtree(str(self.output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def persist(self, test_accuracy: float, input_shape: list):
        # TODO: for now we persist input_shape but it's not ideal.
        # will require refactoring.
        if test_accuracy > self.best_test_accuracy:  # best.
            self.best_test_accuracy = test_accuracy
            print('Best test accuracy reached. Saving model.')
            save_weights(self.net, str(self.output_dir / f'best_model_{self.best_test_accuracy:.4f}.h5'), input_shape)
            save_weights(self.net, str(self.output_dir / 'best_model.h5'), input_shape)
        save_weights(self.net, str(self.output_dir / 'final_model.h5'), input_shape)
