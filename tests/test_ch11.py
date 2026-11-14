"""Tests for Chapter 11: tinyml.autograd, tinyml.nn, and tinyml.optim.

Run from Instructor Materials/Labs:  python -m unittest tests.test_ch11
"""
import unittest

import numpy as np

from tinyml.autograd import Tensor, check_grad, unbroadcast
from tinyml.nn import (MLP, Linear, Module, ReLU, Sequential, cross_entropy,
                       log_softmax, mse_loss, softmax)
from tinyml.optim import SGD, Adam, clip_grad_norm

TOL = 1e-6


def t(rng, *shape, positive=False):
    data = rng.normal(size=shape)
    if positive:
        data = np.abs(data) + 0.5
    return Tensor(data, requires_grad=True)


class TestGradients(unittest.TestCase):
    """Finite-difference checks of every Tensor operation."""

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def check(self, f, inputs):
        # Weight the output by fixed random numbers so every element matters.
        out_shape = f(inputs).shape
        w = np.random.default_rng(1).normal(size=out_shape)
        err = check_grad(lambda xs: (f(xs) * w).sum(), inputs)
        self.assertLess(err, TOL)

    def test_add_broadcast(self):
        self.check(lambda v: v[0] + v[1], [t(self.rng, 4, 3), t(self.rng, 3)])
        self.check(lambda v: v[0] + v[1], [t(self.rng, 4, 1), t(self.rng, 1, 3)])
        self.check(lambda v: v[0] + 2.0, [t(self.rng, 2, 3)])
        self.check(lambda v: 2.0 + v[0], [t(self.rng, 2, 3)])

    def test_sub_neg(self):
        self.check(lambda v: v[0] - v[1], [t(self.rng, 4, 3), t(self.rng, 4, 1)])
        self.check(lambda v: 1.0 - v[0], [t(self.rng, 3)])
        self.check(lambda v: -v[0], [t(self.rng, 3)])

    def test_mul_div(self):
        self.check(lambda v: v[0] * v[1], [t(self.rng, 2, 3), t(self.rng, 3)])
        self.check(lambda v: v[0] / v[1],
                   [t(self.rng, 2, 3), t(self.rng, 2, 1, positive=True)])
        self.check(lambda v: 3.0 / v[0], [t(self.rng, 4, positive=True)])
        self.check(lambda v: 3.0 * v[0], [t(self.rng, 4)])

    def test_pow(self):
        self.check(lambda v: v[0] ** 2, [t(self.rng, 3, 2)])
        self.check(lambda v: v[0] ** 0.5, [t(self.rng, 3, 2, positive=True)])
        self.check(lambda v: v[0] ** -1.5, [t(self.rng, 3, positive=True)])

    def test_matmul(self):
        self.check(lambda v: v[0] @ v[1], [t(self.rng, 4, 3), t(self.rng, 3, 2)])
        # Batched, with the second operand broadcast over the batch.
        self.check(lambda v: v[0] @ v[1], [t(self.rng, 2, 4, 3), t(self.rng, 3, 5)])
        self.check(lambda v: v[0] @ v[1],
                   [t(self.rng, 2, 2, 4, 3), t(self.rng, 2, 2, 3, 4)])

    def test_sum_mean(self):
        x = [t(self.rng, 3, 4, 2)]
        self.check(lambda v: v[0].sum(), x)
        self.check(lambda v: v[0].sum(axis=1), x)
        self.check(lambda v: v[0].sum(axis=(0, 2), keepdims=True), x)
        self.check(lambda v: v[0].mean(), x)
        self.check(lambda v: v[0].mean(axis=-1), x)
        self.check(lambda v: v[0].mean(axis=(0, 2)), x)

    def test_max(self):
        x = [t(self.rng, 3, 4, 2)]
        self.check(lambda v: v[0].max(), x)
        self.check(lambda v: v[0].max(axis=1), x)
        self.check(lambda v: v[0].max(axis=(1, 2), keepdims=True), x)

    def test_max_ties_split_gradient(self):
        x = Tensor([1.0, 3.0, 3.0], requires_grad=True)
        x.max().backward()
        np.testing.assert_allclose(x.grad, [0.0, 0.5, 0.5])

    def test_shape_ops(self):
        x = [t(self.rng, 2, 3, 4)]
        self.check(lambda v: v[0].reshape(6, 4), x)
        self.check(lambda v: v[0].reshape((4, -1)), x)
        self.check(lambda v: v[0].transpose(2, 0, 1), x)
        self.check(lambda v: v[0].transpose(), x)
        self.check(lambda v: v[0][:, 1:, ::2], x)
        self.check(lambda v: v[0].T @ v[0], [t(self.rng, 4, 6)])

    def test_fancy_index_repeats(self):
        w = t(self.rng, 5, 3)
        idx = np.array([0, 2, 2, 4, 0, 0])
        self.check(lambda v: v[0][idx], [w])
        self.check(lambda v: v[0][np.arange(3), np.array([1, 0, 2])], [t(self.rng, 3, 3)])
        self.check(lambda v: v[0][Tensor([1, 1])], [w])

    def test_elementwise(self):
        x = [t(self.rng, 3, 4)]
        self.check(lambda v: v[0].exp(), x)
        self.check(lambda v: v[0].tanh(), x)
        self.check(lambda v: v[0].sigmoid(), x)
        self.check(lambda v: v[0].relu(), x)
        self.check(lambda v: v[0].log(), [t(self.rng, 3, 4, positive=True)])

    def test_reused_node_and_deep_chain(self):
        x = t(self.rng, 3)
        self.check(lambda v: (v[0] * v[0] + v[0]).tanh() * v[0], [x])
        # A chain deeper than Python's recursion limit.
        y = Tensor(0.5, requires_grad=True)
        z = y
        for _ in range(3000):
            z = z * 1.0
        z.backward()
        self.assertAlmostEqual(y.grad.item(), 1.0)


class TestEngine(unittest.TestCase):
    def test_unbroadcast(self):
        g = np.ones((4, 2, 3))
        self.assertEqual(unbroadcast(g, (3,)).shape, (3,))
        self.assertEqual(unbroadcast(g, (2, 1)).shape, (2, 1))
        np.testing.assert_allclose(unbroadcast(g, (2, 1)), np.full((2, 1), 12.0))

    def test_known_gradient(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        (x * x).sum().backward()
        np.testing.assert_allclose(x.grad, [2.0, 4.0, 6.0])

    def test_leaf_gradients_accumulate(self):
        x = Tensor(2.0, requires_grad=True)
        (x * 3.0).backward()
        (x * 3.0).backward()
        self.assertAlmostEqual(x.grad.item(), 6.0)

    def test_constants_get_no_grad(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0], requires_grad=True)
        (a * b).sum().backward()
        self.assertIsNone(a.grad)
        self.assertFalse((a + 1.0).requires_grad)

    def test_ndarray_on_left(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = np.array([3.0, 4.0]) * x
        self.assertIsInstance(y, Tensor)
        y.sum().backward()
        np.testing.assert_allclose(x.grad, [3.0, 4.0])

    def test_sigmoid_is_stable(self):
        x = Tensor([-1000.0, 0.0, 1000.0])
        np.testing.assert_allclose(x.sigmoid().data, [0.0, 0.5, 1.0])


class TestNN(unittest.TestCase):
    def test_parameter_discovery(self):
        class Net(Module):
            def __init__(self):
                self.blocks = [Linear(2, 3, 0), (Linear(3, 3, 1), ReLU())]
                self.head = Linear(3, 1, 2)
                self.shared = self.head.weight          # counted once
                self.const = Tensor(np.ones(3))         # not trainable
        net = Net()
        self.assertEqual(len(net.parameters()), 6)
        self.assertEqual(net.num_parameters(), 9 + 12 + 4)
        net.eval()
        self.assertFalse(net.blocks[1][0].training)
        net.train()
        self.assertTrue(net.head.training)

    def test_mlp_shapes_and_seed(self):
        a = MLP([4, 8, 3], rng=0)
        b = MLP([4, 8, 3], rng=0)
        x = Tensor(np.ones((5, 4)))
        self.assertEqual(a(x).shape, (5, 3))
        np.testing.assert_allclose(a(x).data, b(x).data)
        self.assertEqual(len(a.parameters()), 4)

    def test_linear_init_scale(self):
        layer = Linear(1000, 500, rng=0)
        self.assertAlmostEqual(layer.weight.data.std(), np.sqrt(2 / 1000), places=3)
        layer = Linear(1000, 500, rng=0, init="glorot")
        self.assertAlmostEqual(layer.weight.data.std(), np.sqrt(2 / 1500), places=3)

    def test_cross_entropy_known_and_stable(self):
        logits = Tensor(np.zeros((2, 4)))
        self.assertAlmostEqual(cross_entropy(logits, [0, 3]).item(), np.log(4))
        big = Tensor([[1000.0, 0.0], [0.0, 1000.0]])
        self.assertAlmostEqual(cross_entropy(big, [0, 1]).item(), 0.0)
        self.assertAlmostEqual(cross_entropy(big, [1, 1]).item(), 500.0)

    def test_cross_entropy_gradient(self):
        rng = np.random.default_rng(0)
        logits = Tensor(rng.normal(size=(5, 3)), requires_grad=True)
        y = np.array([0, 2, 1, 1, 0])
        self.assertLess(check_grad(lambda v: cross_entropy(v[0], y), [logits]), TOL)
        # The analytic gradient is (softmax - onehot) / n.
        logits.grad = None
        cross_entropy(logits, y).backward()
        p = softmax(Tensor(logits.data)).data
        expected = (p - np.eye(3)[y]) / 5
        np.testing.assert_allclose(logits.grad, expected, atol=1e-12)

    def test_sequence_cross_entropy_and_softmax(self):
        rng = np.random.default_rng(0)
        logits = Tensor(rng.normal(size=(2, 3, 4)), requires_grad=True)
        y = rng.integers(0, 4, size=(2, 3))
        self.assertLess(check_grad(lambda v: cross_entropy(v[0], y), [logits]), TOL)
        np.testing.assert_allclose(softmax(logits).data.sum(axis=-1), 1.0)
        np.testing.assert_allclose(np.exp(log_softmax(logits).data).sum(-1), 1.0)

    def test_mlp_gradient(self):
        net = MLP([3, 4, 2], rng=0, activation="tanh")
        x = np.random.default_rng(1).normal(size=(6, 3))
        y = np.array([0, 1, 1, 0, 1, 0])
        err = check_grad(lambda ps: cross_entropy(net(Tensor(x)), y), net.parameters())
        self.assertLess(err, TOL)

    def test_mse(self):
        pred = Tensor([1.0, 2.0], requires_grad=True)
        loss = mse_loss(pred, np.array([0.0, 0.0]))
        self.assertAlmostEqual(loss.item(), 2.5)
        loss.backward()
        np.testing.assert_allclose(pred.grad, [1.0, 2.0])


class TestOptim(unittest.TestCase):
    def quadratic(self, opt_cls, steps, **kw):
        p = Tensor([5.0, -3.0], requires_grad=True)
        opt = opt_cls([p], **kw)
        for _ in range(steps):
            opt.zero_grad()
            (p * p).sum().backward()
            opt.step()
        return p.data

    def test_sgd_one_step(self):
        p = Tensor([1.0], requires_grad=True)
        opt = SGD([p], lr=0.1)
        (p * p).sum().backward()
        opt.step()
        np.testing.assert_allclose(p.data, [0.8])

    def test_optimizers_converge(self):
        for cls, kw in [(SGD, dict(lr=0.1)), (SGD, dict(lr=0.05, momentum=0.9)),
                        (Adam, dict(lr=0.1)), (Adam, dict(lr=0.1, weight_decay=0.01,
                                                          decoupled=True))]:
            np.testing.assert_allclose(self.quadratic(cls, 500, **kw), 0.0, atol=1e-3)

    def test_adam_first_step_is_lr(self):
        p = Tensor([5.0, -3.0], requires_grad=True)
        opt = Adam([p], lr=0.1)
        (p * p).sum().backward()
        opt.step()
        np.testing.assert_allclose(p.data, [4.9, -2.9], atol=1e-6)

    def test_clip_grad_norm(self):
        p = Tensor([0.0, 0.0], requires_grad=True)
        p.grad = np.array([3.0, 4.0])
        self.assertAlmostEqual(clip_grad_norm([p], 1.0), 5.0)
        np.testing.assert_allclose(p.grad, [0.6, 0.8])

    def test_train_xor(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        net = MLP([2, 8, 2], rng=0)
        opt = Adam(net.parameters(), lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            cross_entropy(net(Tensor(X)), y).backward()
            opt.step()
        self.assertTrue((net(Tensor(X)).data.argmax(1) == y).all())


if __name__ == "__main__":
    unittest.main()
