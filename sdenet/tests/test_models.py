import unittest

import numpy as np
import tensorflow as tf

from sdenet.models import resnet
from sdenet.models.helpers import set_seed, save_weights, load_weights
from sdenet.models.sdenet import Drift, Diffusion, SDENet, ConcatConv2d

# to remove the warnings, we convert everything to float64. TFLOPS perf :(
tf.keras.backend.set_floatx('float64')


class TestModels(unittest.TestCase):

    def test_concat_conv_2d_layers(self):
        dim = 64
        transpose = False  # transpose=False => down=sampling
        d = ConcatConv2d(dim // 2, transpose=transpose)
        self.assertListEqual(list(d(1, np.ones(shape=(16, 28, 28, dim))).shape), [16, 26, 26, dim // 2])
        self.assertListEqual(list(d(1, np.ones(shape=(16, 24, 24, dim))).shape), [16, 22, 22, dim // 2])

        w = d.get_weights()
        d.set_weights(w)
        w2 = d.get_weights()

        self.assertTrue(np.all([np.all(w[i] == w2[i]) for i in range(len(w))]))

        v1 = d(1, np.zeros(shape=(8, 6, 6, dim)))
        v2 = d(1, np.zeros(shape=(8, 6, 6, dim)))
        np.testing.assert_almost_equal(np.array(v1), np.array(v2))

        d.set_weights([q * 2 for q in d.get_weights()])
        v3 = d(1, np.zeros(shape=(8, 6, 6, dim)))
        self.assertTrue(np.any(np.array(v1) != np.array(v3)))

        # With transpose = True => up-sampling.
        transpose = True  # transpose=False => down=sampling
        d = ConcatConv2d(dim // 2, transpose=transpose)
        self.assertListEqual(list(d(1, np.ones(shape=(16, 28, 28, dim))).shape), [16, 30, 30, dim // 2])
        self.assertListEqual(list(d(1, np.ones(shape=(16, 24, 24, dim))).shape), [16, 26, 26, dim // 2])

        w = d.get_weights()
        d.set_weights(w)
        w2 = d.get_weights()

        self.assertTrue(np.all([np.all(w[i] == w2[i]) for i in range(len(w))]))

        v1 = d(1, np.zeros(shape=(8, 6, 6, dim)))
        v2 = d(1, np.zeros(shape=(8, 6, 6, dim)))
        np.testing.assert_almost_equal(np.array(v1), np.array(v2))

        d.set_weights([q * 2 for q in d.get_weights()])
        v3 = d(1, np.zeros(shape=(8, 6, 6, dim)))
        self.assertTrue(np.any(np.array(v1) != np.array(v3)))

        save_weights(d, 'hello.h5')
        load_weights(d, 'hello.h5')

    def test_model_drift(self):
        dim = 64
        # input.shape = output.shape
        d = Drift(dim=dim)

        self.assertListEqual(list(d(0, np.zeros(shape=(16, 12, 12, dim))).shape), [16, 12, 12, dim])
        self.assertListEqual(list(d(0, np.zeros(shape=(16, 6, 6, dim))).shape), [16, 6, 6, dim])
        self.assertListEqual(list(d(0, np.zeros(shape=(8, 6, 6, dim))).shape), [8, 6, 6, dim])

        w = d.get_weights()
        d.set_weights(w)
        w2 = d.get_weights()

        self.assertTrue(np.all([np.all(w[i] == w2[i]) for i in range(len(w))]))

        v1 = d(1, np.zeros(shape=(8, 6, 6, dim)))
        v2 = d(1, np.zeros(shape=(8, 6, 6, dim)))
        np.testing.assert_almost_equal(np.array(v1), np.array(v2))

        d.set_weights([q * 2 for q in d.get_weights()])
        v3 = d(1, np.zeros(shape=(8, 6, 6, dim)))
        self.assertTrue(np.any(np.array(v1) != np.array(v3)))

        save_weights(d, 'hello.h5')
        load_weights(d, 'hello.h5')

    def test_model_diffusion(self):
        dim = 64
        # input.shape = [batch_size, 1]
        # outputs the diffusion (variance) coefficient.
        d = Diffusion(dim, dim)
        self.assertListEqual(list(d(0, np.zeros(shape=(16, 12, 12, dim))).shape), [16, 1])

        w = d.get_weights()
        d.set_weights(w)
        w2 = d.get_weights()

        self.assertTrue(np.all([np.all(w[i] == w2[i]) for i in range(len(w))]))

        v1 = d(1, np.zeros(shape=(8, 6, 6, dim)))
        v2 = d(1, np.zeros(shape=(8, 6, 6, dim)))
        np.testing.assert_almost_equal(np.array(v1), np.array(v2))

        d.set_weights([q * 2 for q in d.get_weights()])
        v3 = d(1, np.zeros(shape=(8, 6, 6, dim)))
        self.assertTrue(np.any(np.array(v1) != np.array(v3)))

        save_weights(d, 'hello.h5')
        load_weights(d, 'hello.h5')

    def test_sde_net_mnist(self):
        dim = 64
        inputs = np.ones(shape=(16, 28, 28, dim))
        d = SDENet(layer_depth=6, num_classes=10, dim=dim)

        def reproducible_call():
            set_seed()
            return d(inputs)

        self.assertListEqual(list(reproducible_call().shape), [16, 10])

        w = d.get_weights()
        d.set_weights(w)
        w2 = d.get_weights()

        self.assertTrue(np.all([np.all(w[i] == w2[i]) for i in range(len(w))]))

        set_seed()
        v1 = reproducible_call()
        set_seed()
        v2 = reproducible_call()
        np.testing.assert_almost_equal(np.array(v1), np.array(v2))

        d.set_weights([q * 2 for q in d.get_weights()])
        v3 = reproducible_call()
        self.assertTrue(np.any(np.array(v1) != np.array(v3)))

        save_weights(d, 'hello.h5')
        load_weights(d, 'hello.h5')

    def test_res_net_mnist(self):
        d = resnet.ResidualNet()
        inputs = np.ones(shape=(16, 28, 28, 1))

        def reproducible_call():
            set_seed()
            return d(inputs)

        self.assertListEqual(list(reproducible_call().shape), [16, 10])

        w = d.get_weights()
        d.set_weights(w)
        w2 = d.get_weights()

        self.assertTrue(np.all([np.all(w[i] == w2[i]) for i in range(len(w))]))

        set_seed()
        v1 = reproducible_call()
        set_seed()
        v2 = reproducible_call()
        np.testing.assert_almost_equal(np.array(v1), np.array(v2))

        d.set_weights([q * 2 for q in d.get_weights()])
        v3 = reproducible_call()
        self.assertTrue(np.any(np.array(v1) != np.array(v3)))

        save_weights(d, 'hello.h5')
        load_weights(d, 'hello.h5')
