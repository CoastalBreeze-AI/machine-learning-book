"""Tests for Chapter 13: the Transformer layers in tinyml.nn and
make_sort_task in tinyml.data.

Run from Instructor Materials/Labs:  python -m unittest tests.test_ch13
"""
import unittest

import numpy as np

from tinyml.autograd import Tensor, check_grad
from tinyml.data import make_sort_task
from tinyml.nn import (Embedding, LayerNorm, MultiHeadAttention,
                       TransformerBlock, causal_mask,
                       scaled_dot_product_attention, sinusoidal_positions)
from tinyml.optim import Adam


class TestEmbeddingAndNorm(unittest.TestCase):
    def test_embedding_lookup_and_gradient(self):
        emb = Embedding(5, 3, rng=0)
        out = emb(np.array([[1, 1, 4]]))
        self.assertEqual(out.shape, (1, 3, 3))
        np.testing.assert_allclose(out.data[0, 0], emb.weight.data[1])
        out.sum().backward()
        np.testing.assert_allclose(emb.weight.grad[:, 0], [0, 2, 0, 0, 1])

    def test_layernorm_standardizes(self):
        ln = LayerNorm(6)
        x = Tensor(np.random.default_rng(0).normal(3, 5, (2, 4, 6)))
        y = ln(x).data
        np.testing.assert_allclose(y.mean(axis=-1), 0, atol=1e-10)
        np.testing.assert_allclose(y.std(axis=-1), 1, atol=1e-3)

    def test_layernorm_gradient(self):
        rng = np.random.default_rng(1)
        ln = LayerNorm(4)
        ln.gamma.data[:] = rng.normal(size=4)
        x = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
        w = rng.normal(size=(3, 4))
        err = check_grad(lambda t: (ln(t[0]) * w).sum(), [x, ln.gamma, ln.beta])
        self.assertLess(err, 1e-6)

    def test_sinusoidal_positions(self):
        pe = sinusoidal_positions(10, 8)
        self.assertEqual(pe.shape, (10, 8))
        np.testing.assert_allclose(pe[0, 0::2], 0)
        np.testing.assert_allclose(pe[0, 1::2], 1)
        np.testing.assert_allclose(pe[3, 0], np.sin(3))


class TestAttention(unittest.TestCase):
    def test_weights_are_distributions(self):
        rng = np.random.default_rng(0)
        q, k, v = (Tensor(rng.normal(size=(2, 5, 4))) for _ in range(3))
        out, w = scaled_dot_product_attention(q, k, v)
        self.assertEqual(out.shape, (2, 5, 4))
        np.testing.assert_allclose(w.data.sum(axis=-1), 1)

    def test_known_answer_with_identical_keys(self):
        q = Tensor(np.ones((1, 2, 3)))
        k = Tensor(np.ones((1, 4, 3)))
        v = Tensor(np.arange(8.0).reshape(1, 4, 2))
        out, w = scaled_dot_product_attention(q, k, v)
        np.testing.assert_allclose(w.data, 0.25)
        np.testing.assert_allclose(out.data[0, 0], [3.0, 4.0])

    def test_causal_mask_blocks_future(self):
        rng = np.random.default_rng(2)
        mha = MultiHeadAttention(8, 2, causal=True, rng=0)
        x = rng.normal(size=(1, 5, 8))
        mha(Tensor(x))
        self.assertTrue(np.allclose(np.triu(mha.last_weights[0, 0], 1), 0))
        # Changing the last token must not change earlier outputs.
        y1 = mha(Tensor(x)).data
        x[0, -1] += 10.0
        y2 = mha(Tensor(x)).data
        np.testing.assert_allclose(y1[0, :-1], y2[0, :-1], atol=1e-12)
        self.assertEqual(causal_mask(3)[0, 2], -1e9)

    def test_attention_gradient(self):
        rng = np.random.default_rng(3)
        mha = MultiHeadAttention(4, 2, causal=True, rng=0)
        x = Tensor(rng.normal(size=(2, 3, 4)), requires_grad=True)
        w = rng.normal(size=(2, 3, 4))
        params = [x] + mha.parameters()
        err = check_grad(lambda t: (mha(t[0]) * w).sum(), params)
        self.assertLess(err, 1e-5)

    def test_block_shapes_and_parameters(self):
        block = TransformerBlock(8, 2, causal=True, rng=0)
        out = block(Tensor(np.zeros((3, 5, 8))))
        self.assertEqual(out.shape, (3, 5, 8))
        # 2 LayerNorms (2*8 each) + qkv (8*24+24) + out (8*8+8) + MLP.
        expected = 32 + 216 + 72 + (8 * 32 + 32) + (32 * 8 + 8)
        self.assertEqual(block.num_parameters(), expected)

    def test_block_can_learn_to_copy_previous_token(self):
        rng = np.random.default_rng(0)
        emb, pos = Embedding(5, 16, rng), Embedding(6, 16, rng)
        block = TransformerBlock(16, 2, causal=True, rng=rng)
        head = Embedding(5, 16, rng)       # reuse as an output matrix
        params = emb.parameters() + pos.parameters() + block.parameters() \
            + head.parameters()
        opt = Adam(params, lr=0.01)
        from tinyml.nn import cross_entropy
        for _ in range(150):
            ids = rng.integers(0, 5, size=(32, 6))
            h = block(emb(ids) + pos(np.arange(6)))
            logits = h @ head.weight.T
            loss = cross_entropy(logits[:, 1:, :], ids[:, :-1])
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.3)


class TestSortTask(unittest.TestCase):
    def test_shapes_and_sorting(self):
        X, Y = make_sort_task(50, length=7, vocab=4, seed=1)
        self.assertEqual(X.shape, (50, 7))
        self.assertTrue(np.all(np.diff(Y, axis=1) >= 0))
        np.testing.assert_array_equal(np.sort(X, axis=1), Y)
        self.assertTrue(X.min() >= 0 and X.max() <= 3)

    def test_seeded(self):
        a, _ = make_sort_task(10, seed=5)
        b, _ = make_sort_task(10, seed=5)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
