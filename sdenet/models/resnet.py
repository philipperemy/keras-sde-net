from tensorflow.keras.layers import Conv2D, ReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from sdenet.models.helpers import fc, norm, conv3x3


class ResidualNet(Model):

    def __init__(self):
        super().__init__()
        self.downsampling_layers = [
            Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same'), norm(64), ReLU(),
            Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding='same'), norm(64), ReLU(),
            Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding='same')
        ]

        self.feature_layers = [ResidualBlock(64, 64) for _ in range(6)]

        self.fc_layers = [
            norm(64),
            ReLU(),
            GlobalAvgPool2D(),
            Flatten(),
            fc(10)
        ]

    def __call__(self, x, training=True):
        for layer in self.downsampling_layers:
            x = layer(x)
        for layer in self.feature_layers:
            x = layer(x)
        for layer in self.fc_layers:
            x = layer(x)
        return x

    def call(self, inputs, training=None, mask=None):
        return self.__call__(inputs, training)

    def get_config(self):
        # not so right but it's ok.
        super().get_config()


class ResidualBlock(Layer):

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = norm(inplanes)
        self.conv1 = conv3x3(planes, stride=stride)
        self.relu = ReLU()
        self.downsample = downsample
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes)

    def __call__(self, x, training=None):
        shortcut = x
        out = self.relu(self.norm1(x))
        if self.downsample is not None:
            shortcut = self.downsample(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + shortcut
