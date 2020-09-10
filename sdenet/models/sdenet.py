#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Philippe Remy <philipperemy>
"""
import math

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from sdenet.models.helpers import norm, fc


class ConcatConv2d(Model):

    def __init__(self, dim_out, ksize=3, stride=1, padding=0, dilation=1, bias=True, transpose=False):
        super().__init__()
        # TODO: no groups in tensorflow/keras.
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
        module = Conv2DTranspose if transpose else Conv2D
        layers = []
        if padding > 0:
            layers.append(ZeroPadding2D(padding=(padding, padding)))
        layers.append(module(
            dim_out, kernel_size=ksize, strides=(stride, stride), padding='valid',
            dilation_rate=(dilation, dilation), use_bias=bias
        ))
        self.m = Sequential(layers)

    def __call__(self, t, x):
        # Add the time in another channel.
        tt = K.ones_like(x[:, :, :, :1]) * t
        ttx = K.concatenate([tt, x], axis=-1)
        return self.m(ttx)


class Drift(Model):

    def __init__(self, dim):
        super().__init__()
        self.norm1 = norm(dim)
        self.relu = ReLU()
        self.conv1 = ConcatConv2d(dim, 3, 1, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, 3, 1, 1, 1)
        self.norm3 = norm(dim)

    def __call__(self, t, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


# https://stackoverflow.com/questions/52622518/how-to-convert-pytorch-adaptive-avg-pool2d-method-to-keras-or-tensorflow
class Diffusion(Model):
    def __init__(self, dim_in, dim_out, task='mnist'):
        super().__init__()
        self.task = task
        self.norm1 = norm(dim_in)
        self.relu = ReLU()
        self.conv1 = ConcatConv2d(dim_out, 3, 1, 1)
        self.norm2 = norm(dim_in)
        self.conv2 = ConcatConv2d(dim_out, 3, 1, 1)
        if self.task == 'svhn':  # SVHN -> Add layer #3
            self.norm3 = norm(dim_in)
            self.conv3 = ConcatConv2d(dim_out, 3, 1, 1)

        self.fc = Sequential([
            norm(dim_out), ReLU(), GlobalAvgPool2D(), Flatten(), fc(1, activation='sigmoid')
        ])

    def __call__(self, t, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        if self.task == 'svhn':  # SVHN -> Add layer #3
            out = self.norm3(out)
            out = self.relu(out)
            out = self.conv3(t, out)
        out = self.fc(out)
        return out


# https://stackoverflow.com/questions/55694721/how-to-specify-padding-with-keras-in-conv2d-layer
# https://github.com/eweill/keras-deepcv/blob/master/models/classification/alexnet.py
class SDENet(Model):
    def __init__(self, layer_depth, num_classes=10, dim=64, task='mnist'):
        super().__init__()
        self.task = task
        self.layer_depth = layer_depth
        self.downsampling_layers = Sequential([
            Conv2D(filters=dim, kernel_size=3, strides=(1, 1), padding='valid'),
            norm(dim),
            ReLU(),
            ZeroPadding2D(padding=(1, 1)),
            Conv2D(filters=dim, kernel_size=4, strides=(2, 2), padding='valid'),
            norm(dim),
            ReLU(),
            ZeroPadding2D(padding=(1, 1)),
            Conv2D(filters=dim, kernel_size=4, strides=(2, 2), padding='valid')
        ])
        self.drift = Drift(dim)
        self.diffusion = Diffusion(dim, dim, task)
        self.fc_layers = Sequential([
            norm(dim),
            ReLU(),
            GlobalAvgPool2D(),
            Flatten(),
            fc(num_classes)
        ])
        self.deltat = 6. / self.layer_depth
        self.sigma = 50

    def __call__(self, x, training_diffusion=False):
        out = self.downsampling_layers(x)
        # assert list(out.shape[1:]) == [6, 6, 64]
        if not training_diffusion:
            t = 0
            diffusion_term = self.sigma * self.diffusion(t, out)
            # assert list(diffusion_term.shape[1:]) == [1]
            diffusion_term = K.expand_dims(diffusion_term, 2)
            diffusion_term = K.expand_dims(diffusion_term, 3)
            # assert list(diffusion_term.shape[1:]) == [1, 1, 1]
            # assert list(self.drift(t, out).shape[1:]) == [6, 6, 64]
            for i in range(self.layer_depth):
                t = 6 * (float(i)) / self.layer_depth
                out = out + self.drift(t, out) * self.deltat + diffusion_term * math.sqrt(
                    self.deltat) * K.random_normal(K.shape(out))
            final_out = self.fc_layers(out)
        else:
            t = 0
            final_out = self.diffusion(t, out)
        return final_out

    def set_sigma(self, sigma):
        print(f'Set net.sigma to {sigma}.')
        self.sigma = sigma
