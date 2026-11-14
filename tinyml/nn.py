"""Neural network layers and losses built on tinyml.autograd (Chapters 11-13).

A Module holds parameters (Tensors with requires_grad=True) and submodules
as ordinary attributes. parameters() finds them by searching the attributes,
including inside lists and tuples, so a new layer only has to assign them.
"""
import numpy as np

from tinyml.autograd import Tensor


def _rng(rng):
    """Accepts None, an integer seed, or a Generator; returns a Generator."""
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


class Module:
    """Base class for layers and models. Subclasses implement forward()."""

    training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _children(self):
        """Yields every attribute value, looking inside lists and tuples."""
        stack = list(reversed(list(vars(self).values())))
        while stack:
            value = stack.pop()
            if isinstance(value, (list, tuple)):
                stack.extend(reversed(value))
            else:
                yield value

    def modules(self):
        """Yields this module and every module inside it, depth first."""
        yield self
        for child in self._children():
            if isinstance(child, Module):
                yield from child.modules()

    def parameters(self):
        """Returns the trainable Tensors, each once, in a fixed order."""
        params, seen = [], set()
        for module in self.modules():
            for child in module._children():
                if (isinstance(child, Tensor) and child.requires_grad
                        and id(child) not in seen):
                    seen.add(id(child))
                    params.append(child)
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def train(self, mode=True):
        """Sets training mode, which changes Dropout and BatchNorm."""
        for module in self.modules():
            module.training = mode
        return self

    def eval(self):
        return self.train(False)

    def num_parameters(self):
        return sum(p.size for p in self.parameters())


class Linear(Module):
    """An affine layer, y = x W + b, for inputs of shape (..., in_features).

    init="he" draws weights with variance 2 / in_features, suited to ReLU;
    init="glorot" uses variance 2 / (in_features + out_features), suited to
    tanh and sigmoid. Biases start at zero.
    """

    def __init__(self, in_features, out_features, rng=None, init="he", bias=True):
        rng = _rng(rng)
        if init == "he":
            std = np.sqrt(2.0 / in_features)
        elif init == "glorot":
            std = np.sqrt(2.0 / (in_features + out_features))
        else:
            raise ValueError(f"unknown init {init!r}")
        self.weight = Tensor(rng.normal(0.0, std, (in_features, out_features)),
                             requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x):
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Sequential(Module):
    """Applies layers in order."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(Sequential):
    """A multilayer perceptron. sizes=[2, 16, 16, 3] means 2 inputs, two
    hidden layers of 16 units, and 3 outputs. The last layer is linear."""

    def __init__(self, sizes, rng=None, activation="relu"):
        rng = _rng(rng)
        act = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid}[activation]
        init = "he" if activation == "relu" else "glorot"
        layers = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(Linear(n_in, n_out, rng, init=init))
            if i < len(sizes) - 2:
                layers.append(act())
        super().__init__(*layers)


def log_softmax(logits, axis=-1):
    """log softmax(z) = z - log sum exp(z), computed without overflow.

    Subtracting the row maximum first changes nothing mathematically but
    keeps exp() from overflowing. The maximum is treated as a constant.
    """
    shifted = logits - logits.data.max(axis=axis, keepdims=True)
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


def softmax(logits, axis=-1):
    """Softmax along axis; rows are nonnegative and sum to 1."""
    shifted = logits - logits.data.max(axis=axis, keepdims=True)
    e = shifted.exp()
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy(logits, y):
    """Mean negative log-likelihood of integer labels y under softmax(logits).

    logits has shape (..., k) and y has the matching shape (...), so the
    same function serves a batch of rows or a batch of sequences.
    """
    k = logits.shape[-1]
    flat = log_softmax(logits.reshape(-1, k))
    y = np.asarray(y, dtype=int).reshape(-1)
    return -flat[np.arange(len(y)), y].mean()


def mse_loss(pred, target):
    """Mean squared error between a Tensor and an array or Tensor."""
    return ((pred - target) ** 2).mean()


from tinyml.base import Estimator   # imported here, where the lab first needs them
from tinyml.optim import Adam


class MLPClassifier(Estimator):
    """An MLP with the Estimator interface, so it fits in a Pipeline.

    Trains with Adam on mini-batches of softmax cross-entropy. hidden=()
    gives a network with no hidden layer: multinomial logistic regression.
    """

    def __init__(self, hidden=(32,), lr=0.01, epochs=50, batch_size=64,
                 weight_decay=0.0, activation="relu", seed=0):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.activation = activation
        self.seed = seed

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_, codes = np.unique(y, return_inverse=True)
        rng = np.random.default_rng(self.seed)
        sizes = [X.shape[1], *self.hidden, len(self.classes_)]
        self.net_ = MLP(sizes, rng=rng, activation=self.activation)
        opt = Adam(self.net_.parameters(), lr=self.lr,
                   weight_decay=self.weight_decay)
        self.loss_curve_ = []
        for epoch in range(self.epochs):
            order = rng.permutation(len(X))
            total = 0.0
            for start in range(0, len(X), self.batch_size):
                batch = order[start:start + self.batch_size]
                opt.zero_grad()
                loss = cross_entropy(self.net_(Tensor(X[batch])), codes[batch])
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            self.loss_curve_.append(total / len(X))
        return self

    def predict_proba(self, X):
        logits = self.net_(Tensor(np.asarray(X, dtype=float)))
        return softmax(logits).data

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def im2col(x, k, stride=1, padding=0):
    """Unfolds every k-by-k patch of x into a column.

    x has shape (N, C, H, W). The result has shape (N, C*k*k, L), where L is
    the number of patch positions, so a convolution becomes one matrix
    product. The backward pass (col2im) adds each column's gradient back to
    the pixels it came from; overlapping patches accumulate.
    """
    N, C, H, W = x.shape
    out_h = (H + 2 * padding - k) // stride + 1
    out_w = (W + 2 * padding - k) // stride + 1
    pad = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    xp = np.pad(x.data, pad)
    cols = np.empty((N, C, k, k, out_h, out_w))
    for di in range(k):                   # k*k strided slices, not N*L loops
        for dj in range(k):
            cols[:, :, di, dj] = xp[:, :, di:di + stride * out_h:stride,
                                    dj:dj + stride * out_w:stride]

    def backward(g):
        g = g.reshape(N, C, k, k, out_h, out_w)
        dxp = np.zeros_like(xp)
        for di in range(k):
            for dj in range(k):
                dxp[:, :, di:di + stride * out_h:stride,
                    dj:dj + stride * out_w:stride] += g[:, :, di, dj]
        return (dxp[:, :, padding:padding + H, padding:padding + W],)

    data = cols.reshape(N, C * k * k, out_h * out_w)
    return Tensor.from_op(data, (x,), backward, "im2col"), (out_h, out_w)


class Conv2d(Module):
    """A 2-D convolution (strictly, a cross-correlation, as in every deep
    learning library). Input (N, C_in, H, W); output (N, C_out, H', W').

    Every output position uses the same weights: that is weight sharing.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, rng=None):
        rng = _rng(rng)
        fan_in = in_channels * kernel_size * kernel_size
        self.k, self.stride, self.padding = kernel_size, stride, padding
        self.weight = Tensor(rng.normal(0.0, np.sqrt(2.0 / fan_in),
                                        (out_channels, fan_in)), requires_grad=True)
        self.bias = Tensor(np.zeros((out_channels, 1)), requires_grad=True)

    def forward(self, x):
        cols, (out_h, out_w) = im2col(x, self.k, self.stride, self.padding)
        out = self.weight @ cols + self.bias      # (F, CKK) @ (N, CKK, L)
        return out.reshape(x.shape[0], -1, out_h, out_w)


class MaxPool2d(Module):
    """Takes the maximum over non-overlapping k-by-k windows. H and W must
    be divisible by k. Has no parameters."""

    def __init__(self, kernel_size=2):
        self.k = kernel_size

    def forward(self, x):
        N, C, H, W = x.shape
        k = self.k
        windows = x.reshape(N, C, H // k, k, W // k, k)
        return windows.max(axis=(3, 5))


class Flatten(Module):
    """Reshapes (N, ...) to (N, everything else), from conv to linear layers."""

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Dropout(Module):
    """Zeros each input with probability p during training (inverted
    dropout: survivors are scaled by 1/(1-p), so eval needs no change)."""

    def __init__(self, p=0.5, rng=None):
        self.p = p
        self.rng = _rng(rng)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * keep


class BatchNorm(Module):
    """Batch normalization for (N, C) or (N, C, H, W) inputs.

    In training mode each channel is standardized with the mean and variance
    of the current batch, and running averages of those statistics are kept.
    In eval mode the running averages are used instead, so a prediction for
    one example doesn't depend on the other examples in its batch.
    """

    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)
        self.running_mean = np.zeros(num_features)   # arrays, not parameters
        self.running_var = np.ones(num_features)
        self.momentum, self.eps = momentum, eps

    def forward(self, x):
        axes = (0,) if x.ndim == 2 else (0, 2, 3)
        shape = (1, -1) if x.ndim == 2 else (1, -1, 1, 1)
        if self.training:
            mean = x.mean(axis=axes, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
            m = self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * mean.data.ravel()
            self.running_var = (1 - m) * self.running_var + m * var.data.ravel()
        else:
            mean = Tensor(self.running_mean.reshape(shape))
            var = Tensor(self.running_var.reshape(shape))
        x_hat = (x - mean) / (var + self.eps) ** 0.5
        return x_hat * self.gamma.reshape(shape) + self.beta.reshape(shape)


class Embedding(Module):
    """A lookup table: row i of weight is the vector for token i.

    Indexing a Tensor with an integer array is differentiable, and repeated
    tokens accumulate gradient, so no special backward pass is needed.
    Input: integer array of shape (...). Output: (..., dim).
    """

    def __init__(self, num_embeddings, dim, rng=None, std=0.02):
        rng = _rng(rng)
        self.weight = Tensor(rng.normal(0.0, std, (num_embeddings, dim)),
                             requires_grad=True)

    def forward(self, ids):
        return self.weight[np.asarray(ids, dtype=int)]


def causal_mask(n):
    """An (n, n) array that is 0 where query i may see key j (j <= i) and a
    large negative number elsewhere. Added to scores before the softmax."""
    return np.triu(np.full((n, n), -1e9), k=1)


def scaled_dot_product_attention(q, k, v, mask=None):
    """softmax(q k^T / sqrt(d_k) + mask) v, batched over leading axes.

    q: (..., n_q, d_k), k: (..., n_k, d_k), v: (..., n_k, d_v).
    Returns (output (..., n_q, d_v), weights (..., n_q, n_k)).
    """
    d_k = q.shape[-1]
    swap_last_two = tuple(range(k.ndim - 2)) + (k.ndim - 1, k.ndim - 2)
    scores = q @ k.transpose(swap_last_two) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


class MultiHeadAttention(Module):
    """Multi-head self-attention for inputs of shape (batch, n, d_model).

    Each head attends in its own d_model / n_heads dimensional subspace.
    The most recent attention weights are kept in last_weights, with shape
    (batch, n_heads, n, n), so that you can inspect them.
    """

    def __init__(self, d_model, n_heads, causal=False, rng=None):
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        rng = _rng(rng)
        self.n_heads, self.d_head, self.causal = n_heads, d_model // n_heads, causal
        self.qkv = Linear(d_model, 3 * d_model, rng, init="glorot")
        self.out = Linear(d_model, d_model, rng, init="glorot")
        self.last_weights = None

    def forward(self, x):
        batch, n, d_model = x.shape
        qkv = self.qkv(x).reshape(batch, n, 3, self.n_heads, self.d_head)
        qkv = qkv.transpose(2, 0, 3, 1, 4)        # (3, batch, heads, n, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        mask = causal_mask(n) if self.causal else None
        heads, weights = scaled_dot_product_attention(q, k, v, mask)
        self.last_weights = weights.data
        merged = heads.transpose(0, 2, 1, 3).reshape(batch, n, d_model)
        return self.out(merged)


def sinusoidal_positions(n_positions, dim):
    """The fixed positional encodings of Vaswani et al. (2017):
    PE[p, 2i] = sin(p / 10000^(2i/dim)), PE[p, 2i+1] = cos(same angle).
    Returns a constant (n_positions, dim) array."""
    positions = np.arange(n_positions)[:, None]
    rates = 1.0 / 10000 ** (np.arange(0, dim, 2) / dim)
    pe = np.zeros((n_positions, dim))
    pe[:, 0::2] = np.sin(positions * rates)
    pe[:, 1::2] = np.cos(positions * rates)[:, : dim // 2]
    return pe


class LayerNorm(Module):
    """Standardizes each vector over its last axis, then rescales.

    Unlike BatchNorm, the statistics come from one example's own features,
    so training and inference behave identically and batch size is irrelevant.
    """

    def __init__(self, dim, eps=1e-5):
        self.gamma = Tensor(np.ones(dim), requires_grad=True)
        self.beta = Tensor(np.zeros(dim), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return (x - mean) / (var + self.eps) ** 0.5 * self.gamma + self.beta


class TransformerBlock(Module):
    """A pre-norm Transformer block:

        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    The MLP widens to d_ff units with a ReLU and projects back to d_model.
    """

    def __init__(self, d_model, n_heads, d_ff=None, causal=False, rng=None):
        rng = _rng(rng)
        d_ff = d_ff or 4 * d_model
        self.norm1, self.norm2 = LayerNorm(d_model), LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal, rng=rng)
        self.mlp = Sequential(Linear(d_model, d_ff, rng), ReLU(),
                              Linear(d_ff, d_model, rng, init="glorot"))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))
