"""Chapter 12 lab: Listings 12.8, 12.9, 12.10.

Generated from the chapter by tools/listings.py, so it matches the printed
code; regenerate it rather than editing it. Run it from this folder or from
the Labs folder, for example: python labs/ch12_shapes_lab.py
"""
import os
import sys

# Make tinyml, in the folder above this one, importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---- Listing 12.8 ----
import numpy as np

from tinyml.autograd import Tensor
from tinyml.nn import cross_entropy
from tinyml.optim import Adam


def augment(X, rng, max_shift=1):
    """Random horizontal flips and shifts of up to max_shift pixels.

    Both leave the class of every shape unchanged, so each epoch the network
    sees slightly different images with the same labels.
    """
    X = X.copy()
    flip = rng.random(len(X)) < 0.5
    X[flip] = X[flip, :, :, ::-1]
    s = max_shift
    padded = np.pad(X, ((0, 0), (0, 0), (s, s), (s, s)))
    H, W = X.shape[2:]
    for i, (dy, dx) in enumerate(rng.integers(0, 2 * s + 1, size=(len(X), 2))):
        X[i] = padded[i, :, dy:dy + H, dx:dx + W]
    return X


def cosine_lr(epoch, epochs, base_lr, warmup=2):
    """Linear warmup for a few epochs, then cosine decay toward zero."""
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, epochs - warmup)
    return 0.5 * base_lr * (1 + np.cos(np.pi * progress))


def accuracy_of(model, X, y, batch=500):
    model.eval()
    hits = sum((model(Tensor(X[i:i + batch])).data.argmax(1) == y[i:i + batch]).sum()
               for i in range(0, len(X), batch))
    return hits / len(X)


# ---- Listing 12.9 ----
def train(model, X, y, X_val, y_val, epochs=20, lr=3e-3, weight_decay=0.0,
          aug=False, patience=5, batch_size=64, seed=0):
    """Mini-batch Adam with a cosine schedule and early stopping.

    After each epoch the model is scored on the validation set. Training
    stops when validation accuracy hasn't improved for `patience` epochs,
    and the weights from the best epoch are restored.
    """
    rng = np.random.default_rng(seed)
    params = model.parameters()
    opt = Adam(params, lr=lr, weight_decay=weight_decay, decoupled=True)
    best_acc, best_weights, best_epoch = -1.0, None, 0
    for epoch in range(epochs):
        opt.lr = cosine_lr(epoch, epochs, lr)
        model.train()
        X_epoch = augment(X, rng) if aug else X
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            idx = order[start:start + batch_size]
            opt.zero_grad()
            cross_entropy(model(Tensor(X_epoch[idx])), y[idx]).backward()
            opt.step()
        val_acc = accuracy_of(model, X_val, y_val)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_weights = [p.data.copy() for p in params]
        elif epoch - best_epoch >= patience:
            break
    for p, saved in zip(params, best_weights):
        p.data[...] = saved
    # BatchNorm running statistics are not parameters; a fuller version
    # would save and restore them too.
    return best_acc, best_epoch + 1, epoch + 1


# ---- Listing 12.10 ----
import time

from tinyml.data import make_shapes
from tinyml.nn import (BatchNorm, Conv2d, Dropout, Flatten, Linear, MaxPool2d,
                       ReLU, Sequential)

X, y = make_shapes(2000, seed=0)
X_train, y_train, X_val, y_val = X[:1600], y[:1600], X[1600:], y[1600:]
X_test, y_test = make_shapes(1000, seed=1)


def mlp(rng, p=0.0):
    return Sequential(Flatten(), Linear(144, 64, rng), ReLU(), Dropout(p, rng),
                      Linear(64, 4, rng))


def cnn(rng, p=0.0):
    return Sequential(
        Conv2d(1, 8, 3, padding=1, rng=rng), BatchNorm(8), ReLU(), MaxPool2d(2),
        Conv2d(8, 16, 3, padding=1, rng=rng), BatchNorm(16), ReLU(), MaxPool2d(2),
        Flatten(), Dropout(p, rng), Linear(16 * 3 * 3, 4, rng))


configs = [
    ("MLP", mlp, {}, {}),
    ("MLP + dropout, decay, aug", mlp, {"p": 0.3}, {"weight_decay": 1e-3, "aug": True}),
    ("CNN", cnn, {}, {}),
    ("CNN + dropout, decay, aug", cnn, {"p": 0.3}, {"weight_decay": 1e-3, "aug": True}),
]
for name, build, arch, opts in configs:
    rng = np.random.default_rng(0)
    model = build(rng, **arch)
    start = time.time()
    val, best, ran = train(model, X_train, y_train, X_val, y_val, epochs=25, **opts)
    test = accuracy_of(model, X_test, y_test)
    print(f"{name:26s} params {model.num_parameters():5d}  train "
          f"{accuracy_of(model, X_train, y_train):.3f}  val {val:.3f}  test {test:.3f}"
          f"  best epoch {best:2d}/{ran:2d}  {time.time() - start:4.0f} s")
