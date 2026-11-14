"""Chapter 13 lab: Listings 13.8, 13.9, 13.10, 13.11.

Generated from the chapter by tools/listings.py, so it matches the printed
code; regenerate it rather than editing it. Run it from this folder or from
the Labs folder, for example: python labs/ch13_sort_lab.py
"""
import os
import sys

# Make tinyml, in the folder above this one, importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---- Listing 13.8 ----
import numpy as np
from tinyml.autograd import Tensor
from tinyml.nn import (Module, Embedding, Linear, LayerNorm, TransformerBlock,
                       sinusoidal_positions, cross_entropy)

SEP = 10   # the separator token; digits are 0-9, so the vocabulary has 11


class SortTransformer(Module):
    """A decoder-only Transformer that reads [x_1..x_L, SEP, y_1..y_L-1]
    and predicts the next token at every position.

    positions is "learned", "sinusoidal", or "none" (for the ablation).
    """

    def __init__(self, length=6, d_model=32, n_heads=2, n_layers=2,
                 positions="learned", seed=0):
        rng = np.random.default_rng(seed)
        self.length, self.positions = length, positions
        n_ctx = 2 * length
        self.tok = Embedding(SEP + 1, d_model, rng)
        if positions == "learned":
            self.pos = Embedding(n_ctx, d_model, rng)
        elif positions == "sinusoidal":
            self.pos_table = sinusoidal_positions(n_ctx, d_model)
        self.blocks = [TransformerBlock(d_model, n_heads, causal=True, rng=rng)
                       for _ in range(n_layers)]
        self.norm = LayerNorm(d_model)
        self.head = Linear(d_model, SEP + 1, rng, init="glorot")

    def forward(self, ids):
        n = ids.shape[1]
        h = self.tok(ids)                              # (batch, n, d_model)
        if self.positions == "learned":
            h = h + self.pos(np.arange(n))
        elif self.positions == "sinusoidal":
            h = h + self.pos_table[:n]
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))                 # (batch, n, vocab)


def make_batch(X, Y):
    """Inputs [x, SEP, y_1..y_{L-1}] and next-token targets [x_2.., SEP, y]."""
    sep = np.full((len(X), 1), SEP)
    seq = np.concatenate([X, sep, Y], axis=1)          # length 2L + 1
    return seq[:, :-1], seq[:, 1:]


def loss_on_answer(model, X, Y):
    """Cross-entropy on the L answer positions only; the inputs are random,
    so predicting them is impossible and would only add noise."""
    inputs, targets = make_batch(X, Y)
    logits = model(inputs)
    L = model.length
    return cross_entropy(logits[:, L:, :], targets[:, L:])


def generate(model, X):
    """Greedy decoding: append the most probable token L times."""
    seq = np.concatenate([X, np.full((len(X), 1), SEP)], axis=1)
    for _ in range(model.length):
        logits = model(seq).data[:, -1, :]
        seq = np.concatenate([seq, logits.argmax(axis=1)[:, None]], axis=1)
    return seq[:, -model.length:]


# ---- Listing 13.9 ----
from tinyml.data import make_sort_task
from tinyml.optim import Adam


def train(positions="learned", task="sort", steps=1500, batch_size=64,
          lr=3e-3, seed=0, verbose=True):
    """Trains on 20,000 examples; reports accuracy on 1,000 new ones."""
    X_train, Y_train = make_sort_task(20_000, seed=seed)
    X_test, Y_test = make_sort_task(1_000, seed=seed + 100)
    if task == "reverse":
        Y_train, Y_test = np.flip(X_train, 1), np.flip(X_test, 1)
    model = SortTransformer(positions=positions, seed=seed)
    optimizer = Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for step in range(1, steps + 1):
        idx = rng.integers(len(X_train), size=batch_size)
        loss = loss_on_answer(model, X_train[idx], Y_train[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose and step % 300 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}")
    pred = generate(model, X_test)
    token_acc = (pred == Y_test).mean()
    seq_acc = (pred == Y_test).all(axis=1).mean()
    print(f"{task:7s} positions={positions:10s} token accuracy {token_acc:.1%}, "
          f"whole-sequence accuracy {seq_acc:.1%} "
          f"({model.num_parameters():,} parameters)")
    return model, X_test, Y_test


# ---- Listing 13.10 ----
def show_attention(model, x, y, layer, head):
    """Prints one head's attention weights, in percent, for one example.

    Rows are the positions that predict the answer tokens; columns are the
    tokens they may attend to. Weights under 5 percent are shown as dots.
    """
    inputs, _ = make_batch(x[None, :], y[None, :])
    model(inputs)                                    # fills last_weights
    weights = model.blocks[layer].attn.last_weights[0, head]
    names = [str(t) if t != SEP else "|" for t in inputs[0]]
    L = model.length
    print(f"layer {layer}, head {head}")
    print("             " + "".join(f"{name:>4}" for name in names))
    for i in range(L, len(names)):
        cells = "".join(f"{round(100 * w):4d}" if w >= 0.05 else "   ."
                        for w in weights[i])
        print(f"  {names[i]} -> y{i - L + 1}:  {cells}")

model, X_test, Y_test = train(positions="learned", task="sort")
print("input", X_test[0], "-> output", generate(model, X_test[:1])[0])
show_attention(model, X_test[0], Y_test[0], layer=0, head=0)
show_attention(model, X_test[0], Y_test[0], layer=0, head=1)


# ---- Listing 13.11 ----
for task, positions in [("sort", "sinusoidal"), ("sort", "none"),
                        ("reverse", "learned"), ("reverse", "none")]:
    train(positions=positions, task=task, verbose=False)
