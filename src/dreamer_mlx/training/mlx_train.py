"""Training loops in Apple MLX, modelled on flowjax.train.

Loss function conventions (analogous to flowjax, but the model replaces the
``(params, static)`` pair, since MLX modules already separate trainable from frozen
parameters via ``Module.freeze()`` / ``Module.trainable_parameters()``):

- ``fit_to_data``:            ``loss_fn(model, *arrays, key=key) -> scalar``
- ``fit_to_key_based_loss``:  ``loss_fn(model, key=key) -> scalar``
"""

from collections.abc import Callable, Sequence
from functools import partial

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def train_val_split(key: mx.array, arrays: Sequence[mx.array], val_prop: float = 0.1):
    """Random train/validation split of arrays with matching size on axis 0."""
    if not 0 <= val_prop <= 1:
        raise ValueError("val_prop should be between 0 and 1.")
    n = arrays[0].shape[0]
    if not all(a.shape[0] == n for a in arrays):
        raise ValueError("Array dimensions must match along axis 0.")

    n_train = n - round(val_prop * n)
    perm = mx.random.permutation(n, key=key)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    return [a[train_idx] for a in arrays], [a[val_idx] for a in arrays]


def _add_batch(arr: mx.array, batch_size: int) -> mx.array:
    """Add a leading batch dimension, dropping the last batch if truncated."""
    batch_size = min(batch_size, arr.shape[0])
    n_batches = arr.shape[0] // batch_size
    return arr[: n_batches * batch_size].reshape(n_batches, batch_size, *arr.shape[1:])


def get_batches(arrays: Sequence[mx.array], batch_size: int):
    """Reshape ``(n, *rest)`` arrays to ``(n_batches, batch_size, *rest)``.

    The last batch is dropped if truncated (keeps shapes fixed, avoiding recompilation).
    """
    n = arrays[0].shape[0]
    if not all(a.shape[0] == n for a in arrays):
        raise ValueError("Array dimensions do not match along the batch axis.")
    return tuple(_add_batch(a, batch_size) for a in arrays)


def count_fruitless(losses: list[float]) -> int:
    """Number of epochs since the minimum loss."""
    min_idx = min(range(len(losses)), key=losses.__getitem__)
    return len(losses) - min_idx - 1


def _make_step(model: nn.Module, optimizer: optim.Optimizer, loss_fn: Callable):
    """Build a compiled training step ``step(key, *batch) -> loss``.

    ``nn.value_and_grad`` differentiates w.r.t. ``model.trainable_parameters()`` only.
    ``mx.compile`` needs the model/optimizer/RNG state declared as inputs and outputs,
    as these are updated in place.
    """
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(key, *batch):
        loss, grads = loss_and_grad(model, *batch, key=key)
        optimizer.update(model, grads)
        return loss

    return step, state


# --------------------------------------------------------------------------- #
# Training loops
# --------------------------------------------------------------------------- #
def fit_to_key_based_loss(
    key: mx.array,
    model: nn.Module,
    *,
    loss_fn: Callable[..., mx.array],
    steps: int,
    learning_rate: float = 5e-4,
    optimizer: optim.Optimizer | None = None,
    show_progress: bool = True,
):
    """Train a model with a loss of the form ``loss_fn(model, key=key)``.

    E.g. for variational objectives such as the ELBO.

    Returns:
        A tuple containing the trained model (trained in place) and the losses.
    """
    optimizer = optimizer if optimizer is not None else optim.Adam(learning_rate)
    optimizer.init(model.trainable_parameters())
    step, state = _make_step(model, optimizer, loss_fn)

    losses = []
    keys = tqdm(mx.random.split(key, steps), total=steps, disable=not show_progress)
    for k in keys:
        loss = step(k)
        mx.eval(state, loss)
        losses.append(loss.item())
        keys.set_postfix({"loss": losses[-1]})
    return model, losses


def fit_to_data(
    key: mx.array,
    model: nn.Module,
    data: mx.array | Sequence[mx.array] = (),
    *,
    loss_fn: Callable[..., mx.array],
    learning_rate: float = 5e-4,
    optimizer: optim.Optimizer | None = None,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    val_prop: float = 0.1,
    return_best: bool = True,
    show_progress: bool = True,
):
    """Train a model on data with early stopping on a held-out validation set.

    Args:
        key: MLX random key (``mx.random.key(seed)``).
        model: The ``nn.Module`` to train (updated in place). Freeze anything that
            shouldn't be trained with ``model.freeze()`` / ``module.freeze()``.
        data: An array, or sequence of arrays, passed positionally to ``loss_fn``
            (e.g. ``(x, condition)``).
        loss_fn: ``loss_fn(model, *arrays, key=key) -> scalar``.
        learning_rate: Adam learning rate. Ignored if ``optimizer`` is given.
        optimizer: An ``mlx.optimizers`` optimizer. Defaults to Adam.
        max_epochs: Maximum number of epochs.
        max_patience: Epochs without validation improvement before stopping.
        batch_size: Batch size. The last batch of each epoch is dropped if truncated.
        val_prop: Proportion of data used for validation.
        return_best: If True, restore the parameters with the lowest validation loss.
        show_progress: Whether to show a progress bar.

    Returns:
        A tuple containing the trained model and a dict of ``{"train", "val"}`` losses.
    """
    data = (data,) if isinstance(data, mx.array) else data
    data = tuple(mx.array(a) for a in data)

    optimizer = optimizer if optimizer is not None else optim.Adam(learning_rate)
    optimizer.init(model.trainable_parameters())
    step, state = _make_step(model, optimizer, loss_fn)

    @partial(mx.compile, inputs=model.state)
    def eval_step(key, *batch):
        return loss_fn(model, *batch, key=key)

    key, subkey = mx.random.split(key)
    train_data, val_data = train_val_split(subkey, data, val_prop=val_prop)
    has_val = val_data[0].shape[0] > 0
    losses = {"train": [], "val": []}

    # MLX arrays are immutable and updates create new arrays, so holding a reference
    # to the parameter tree is a safe snapshot.
    best_params = model.parameters()

    loop = tqdm(range(max_epochs), disable=not show_progress)
    for _ in loop:
        key, k_train, k_val = mx.random.split(key, 3)

        # Shuffle (same permutation applied to all arrays to keep them aligned)
        perm = mx.random.permutation(train_data[0].shape[0], key=k_train)
        train_data = [a[perm] for a in train_data]
        if has_val:
            perm = mx.random.permutation(val_data[0].shape[0], key=k_val)
            val_data = [a[perm] for a in val_data]

        # Train epoch
        model.train()
        batch_losses = []
        for batch in zip(*get_batches(train_data, batch_size), strict=True):
            key, subkey = mx.random.split(key)
            loss = step(subkey, *batch)
            mx.eval(state, loss)
            batch_losses.append(loss)
        losses["train"].append(mx.stack(batch_losses).mean().item())

        # Val epoch
        if has_val:
            model.eval()
            batch_losses = []
            for batch in zip(*get_batches(val_data, batch_size), strict=True):
                key, subkey = mx.random.split(key)
                loss = eval_step(subkey, *batch)
                mx.eval(loss)
                batch_losses.append(loss)
            losses["val"].append(mx.stack(batch_losses).mean().item())
        else:  # no validation data: monitor the training loss
            losses["val"].append(losses["train"][-1])

        loop.set_postfix({k: v[-1] for k, v in losses.items()})
        if losses["val"][-1] == min(losses["val"]):
            best_params = model.parameters()
        elif count_fruitless(losses["val"]) > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    if return_best:
        model.update(best_params)
        mx.eval(model.parameters())
    return model, losses


# --------------------------------------------------------------------------- #
# Example losses and usage
# --------------------------------------------------------------------------- #
def maximum_likelihood_loss(model, x, condition=None, *, key=None):
    """Negative log likelihood. Assumes ``model.log_prob(x, condition)`` exists."""
    return -model.log_prob(x, condition).mean()


if __name__ == "__main__":
    import math

    class DiagGaussian(nn.Module):
        """Toy unconditional model, to demonstrate the API."""

        def __init__(self, dim: int):
            super().__init__()
            self.loc = mx.zeros(dim)
            self.log_scale = mx.zeros(dim)

        def log_prob(self, x, condition=None):
            z = (x - self.loc) * mx.exp(-self.log_scale)
            return (-0.5 * z**2 - self.log_scale - 0.5 * math.log(2 * math.pi)).sum(-1)

    key = mx.random.key(0)
    key, k_data, k_fit = mx.random.split(key, 3)
    x = 2.0 + 0.5 * mx.random.normal((5000, 2), key=k_data)

    model = DiagGaussian(2)
    model, losses = fit_to_data(
        k_fit,
        model,
        x,
        loss_fn=maximum_likelihood_loss,
        learning_rate=1e-2,
        max_epochs=50,
    )
    print("loc:", model.loc, "scale:", mx.exp(model.log_scale))
