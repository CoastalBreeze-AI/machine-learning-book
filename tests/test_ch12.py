"""Tests for Chapter 12: convolution, pooling, dropout, batch norm, make_shapes.

Run from Instructor Materials/Labs:  python -m unittest tests.test_ch12
"""
import unittest

import numpy as np

from tinyml.autograd import Tensor, check_grad
from tinyml.data import SHAPES, make_shapes
from tinyml.nn import (BatchNorm, Conv2d, Dropout, Flatten, Linear, MaxPool2d,
                       ReLU, Sequential, cross_entropy, im2col)
from tinyml.optim import Adam

TOL = 1e-6


def naive_conv(x, w, b, k, stride, padding):
    """A direct, loop-based convolution to compare against."""
    N, C, H, W = x.shape
    F = w.shape[0]
    w = w.reshape(F, C, k, k)
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    oh = (H + 2 * padding - k) // stride + 1
    ow = (W + 2 * padding - k) // stride + 1
    out = np.zeros((N, F, oh, ow))
    for n in range(N):
        for f in range(F):
            for i in range(oh):
                for j in range(ow):
                    patch = xp[n, :, i * stride:i * stride + k, j * stride:j * stride + k]
                    out[n, f, i, j] = (patch * w[f]).sum() + b[f, 0]
    return out


def weighted(f, out_shape, seed=1):
    w = np.random.default_rng(seed).normal(size=out_shape)
    return lambda xs: (f(xs) * w).sum()


class TestConv(unittest.TestCase):
    def test_matches_naive(self):
        rng = np.random.default_rng(0)
        for stride, padding in [(1, 0), (1, 1), (2, 1)]:
            conv = Conv2d(2, 3, 3, stride=stride, padding=padding, rng=0)
            conv.bias.data = rng.normal(size=(3, 1))
            x = rng.normal(size=(2, 2, 7, 6))
            out = conv(Tensor(x)).data
            ref = naive_conv(x, conv.weight.data, conv.bias.data, 3, stride, padding)
            np.testing.assert_allclose(out, ref, atol=1e-12)

    def test_im2col_shape(self):
        cols, (oh, ow) = im2col(Tensor(np.zeros((4, 3, 12, 12))), 3, 1, 1)
        self.assertEqual(cols.shape, (4, 27, 144))
        self.assertEqual((oh, ow), (12, 12))

    def test_conv_gradients(self):
        rng = np.random.default_rng(0)
        conv = Conv2d(2, 3, 3, stride=2, padding=1, rng=0)
        x = Tensor(rng.normal(size=(2, 2, 5, 5)), requires_grad=True)
        out_shape = conv(x).shape
        f = weighted(lambda v: conv(v[0]), out_shape)
        self.assertLess(check_grad(f, [x, conv.weight, conv.bias]), TOL)


class TestLayers(unittest.TestCase):
    def test_maxpool_values_and_gradient(self):
        x = Tensor(np.arange(16.0).reshape(1, 1, 4, 4), requires_grad=True)
        out = MaxPool2d(2)(x)
        np.testing.assert_allclose(out.data[0, 0], [[5, 7], [13, 15]])
        rng = np.random.default_rng(0)
        y = Tensor(rng.normal(size=(2, 3, 4, 6)), requires_grad=True)
        f = weighted(lambda v: MaxPool2d(2)(v[0]), (2, 3, 2, 3))
        self.assertLess(check_grad(f, [y]), TOL)

    def test_flatten(self):
        self.assertEqual(Flatten()(Tensor(np.zeros((5, 2, 3, 3)))).shape, (5, 18))

    def test_dropout_train_and_eval(self):
        drop = Dropout(0.25, rng=0)
        x = Tensor(np.ones((200, 50)))
        out = drop(x).data
        self.assertAlmostEqual((out == 0).mean(), 0.25, delta=0.02)
        self.assertAlmostEqual(out.mean(), 1.0, delta=0.03)       # inverted scaling
        np.testing.assert_allclose(np.unique(out), [0.0, 1 / 0.75])
        drop.eval()
        np.testing.assert_allclose(drop(x).data, x.data)

    def test_batchnorm_train_statistics(self):
        rng = np.random.default_rng(0)
        bn = BatchNorm(3)
        x = Tensor(rng.normal(5.0, 2.0, size=(64, 3, 4, 4)))
        out = bn(x).data
        np.testing.assert_allclose(out.mean(axis=(0, 2, 3)), 0.0, atol=1e-10)
        np.testing.assert_allclose(out.var(axis=(0, 2, 3)), 1.0, atol=1e-3)
        np.testing.assert_allclose(bn.running_mean, 0.1 * x.data.mean(axis=(0, 2, 3)))
        self.assertEqual(len(bn.parameters()), 2)

    def test_batchnorm_eval_uses_running_stats(self):
        bn = BatchNorm(2)
        bn.running_mean = np.array([1.0, -1.0])
        bn.running_var = np.array([4.0, 1.0])
        bn.eval()
        out = bn(Tensor([[3.0, 0.0]])).data      # one example is fine in eval
        np.testing.assert_allclose(out, [[2 / np.sqrt(4 + 1e-5), 1 / np.sqrt(1 + 1e-5)]])

    def test_batchnorm_gradients(self):
        rng = np.random.default_rng(0)
        for shape in [(6, 3), (4, 2, 3, 3)]:
            bn = BatchNorm(shape[1])
            bn.gamma.data = rng.normal(size=shape[1])
            x = Tensor(rng.normal(size=shape), requires_grad=True)
            f = weighted(lambda v: bn(v[0]), shape)
            self.assertLess(check_grad(f, [x, bn.gamma, bn.beta]), TOL)


class TestShapes(unittest.TestCase):
    def test_make_shapes(self):
        X, y = make_shapes(100, size=12, seed=0)
        self.assertEqual(X.shape, (100, 1, 12, 12))
        self.assertEqual(np.bincount(y).tolist(), [25, 25, 25, 25])
        self.assertEqual(len(SHAPES), 4)
        X2, y2 = make_shapes(100, size=12, seed=0)
        np.testing.assert_array_equal(X, X2)
        np.testing.assert_array_equal(y, y2)
        Xc, yc = make_shapes(40, size=16, seed=3, noise=0.0)
        self.assertEqual(Xc.shape, (40, 1, 16, 16))
        self.assertGreaterEqual(Xc.min(), 0.0)          # no noise: shape on black
        self.assertLessEqual(Xc.max(), 1.0)
        self.assertTrue((Xc.reshape(40, -1) > 0).sum(axis=1).min() > 5)

    def test_small_cnn_learns(self):
        X, y = make_shapes(400, seed=0)
        rng = np.random.default_rng(0)
        net = Sequential(Conv2d(1, 8, 3, padding=1, rng=rng), ReLU(), MaxPool2d(2),
                         Flatten(), Linear(8 * 36, 4, rng))
        opt = Adam(net.parameters(), lr=3e-3)
        for _ in range(8):
            for s in range(0, 400, 50):
                opt.zero_grad()
                cross_entropy(net(Tensor(X[s:s + 50])), y[s:s + 50]).backward()
                opt.step()
        acc = (net(Tensor(X)).data.argmax(1) == y).mean()
        self.assertGreater(acc, 0.6)


if __name__ == "__main__":
    unittest.main()
