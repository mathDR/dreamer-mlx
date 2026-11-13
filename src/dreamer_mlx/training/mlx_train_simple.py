"""Training loops in Apple MLX, using MLX's implicit global RNG.

Seed once up front with ``mx.random.seed(seed)``; no keys are threaded through.

Loss function conventions:

- ``fit_to_data``: ``loss_fn(model, *arrays) -> scalar``
- ``fit_to_loss``: ``loss_fn(model) -> scalar``

Any randomness inside a loss (sampling, dropout, ...) just calls ``mx.random.*``.
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
def train_val_split(arrays: Sequence[mx.array], val_prop: float = 0.1):
    """Random train/validation split of arrays with matching size on axis 0."""
    if not 0 <= val_prop <= 1:
        raise ValueError("val_prop should be between 0 and 1.")
    n = arrays[0].shape[0]
    if not all(a.shape[0] == n for a in arrays):
        raise ValueError("Array dimensions must match along axis 0.")

    n_train = n - round(val_prop * n)
    perm = mx.random.permutation(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    return [a[train_idx] for a in arrays], [a[val_idx] for a in arrays]


def shuffle(arrays: Sequence[mx.array]):
    """Shuffle arrays along axis 0 with a shared permutation."""
    perm = mx.random.permutation(arrays[0].shape[0])
    return [a[perm] for a in arrays]


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
    """Build a compiled training step ``step(*batch) -> loss``.

    ``nn.value_and_grad`` differentiates w.r.t. ``model.trainable_parameters()`` only.

    ``mx.random.state`` MUST be in the compile state: otherwise random numbers drawn
    inside ``loss_fn`` are frozen at trace time and every step reuses the same values.
    """
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(*batch):
        loss, grads = loss_and_grad(model, *batch)
        optimizer.update(model, grads)
        return loss

    return step, state


# --------------------------------------------------------------------------- #
# Training loops
# --------------------------------------------------------------------------- #
def fit_to_loss(
    model: nn.Module,
    *,
    loss_fn: Callable[[nn.Module], mx.array],
    steps: int,
    learning_rate: float = 5e-4,
    optimizer: optim.Optimizer | None = None,
    show_progress: bool = True,
):
    """Train a model by minimising ``loss_fn(model)`` for a fixed number of steps.

    Useful for objectives with no dataset, e.g. variational objectives like the ELBO.

    Returns:
        A tuple containing the trained model (updated in place) and the losses.
    """
    optimizer = optimizer if optimizer is not None else optim.Adam(learning_rate)
    optimizer.init(model.trainable_parameters())
    step, state = _make_step(model, optimizer, loss_fn)

    losses = []
    loop = tqdm(range(steps), disable=not show_progress)
    for _ in loop:
        loss = step()
        mx.eval(state, loss)
        losses.append(loss.item())
        loop.set_postfix({"loss": losses[-1]})
    return model, losses


def fit_to_data(
    model: nn.Module,
    data: mx.array | Sequence[mx.array],
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
        model: The ``nn.Module`` to train (updated in place). Freeze anything that
            shouldn't be trained with ``model.freeze()`` / ``module.freeze()``.
        data: An array, or sequence of arrays, passed positionally to ``loss_fn``
            (e.g. ``(x, condition)``).
        loss_fn: ``loss_fn(model, *arrays) -> scalar``.
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

    # Validation loss may also be stochastic (e.g. Monte Carlo), so RNG is in the state.
    eval_state = [model.state, mx.random.state]

    @partial(mx.compile, inputs=eval_state, outputs=eval_state)
    def eval_step(*batch):
        return loss_fn(model, *batch)

    train_data, val_data = train_val_split(data, val_prop=val_prop)
    has_val = val_data[0].shape[0] > 0
    losses: dict[str,list[float]] = {"train": [], "val": []}

    # MLX arrays are immutable and updates create new arrays, so holding a reference
    # to the parameter tree is a safe snapshot.
    best_params = model.parameters()

    loop = tqdm(range(max_epochs), disable=not show_progress)
    for _ in loop:
        train_data = shuffle(train_data)

        # Train epoch
        model.train()
        batch_losses = []
        for batch in zip(*get_batches(train_data, batch_size), strict=True):
            loss = step(*batch)
            mx.eval(state, loss)
            batch_losses.append(loss)
        losses["train"].append(float(mx.stack(batch_losses).mean().item()))

        # Val epoch
        if has_val:
            val_data = shuffle(val_data)
            model.eval()
            batch_losses = []
            for batch in zip(*get_batches(val_data, batch_size), strict=True):
                loss = eval_step(*batch)
                mx.eval(eval_state, loss)
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
def maximum_likelihood_loss(model, x, condition=None):
    """Negative log likelihood. Assumes ``model.log_prob(x, condition)`` exists."""
    return -model.log_prob(x, condition).mean()


def make_elbo_loss(target: Callable[[mx.array], mx.array], num_samples: int):
    """Negative ELBO. Assumes ``model.sample_and_log_prob(shape)`` exists.

    ``target`` maps a batch of samples ``(n, dim)`` to log densities ``(n,)``.
    """

    def elbo_loss(model):
        samples, log_probs = model.sample_and_log_prob((num_samples,))
        return (log_probs - target(samples)).mean()

    return elbo_loss


if __name__ == "__main__":
    import math

    class DiagGaussian(nn.Module):
        """Toy model, to demonstrate the API."""

        def __init__(self, dim: int):
            super().__init__()
            self.loc = mx.zeros(dim)
            self.log_scale = mx.zeros(dim)

        def log_prob(self, x, condition=None):
            z = (x - self.loc) * mx.exp(-self.log_scale)
            return (-0.5 * z**2 - self.log_scale - 0.5 * math.log(2 * math.pi)).sum(-1)

        def sample_and_log_prob(self, shape):
            eps = mx.random.normal((*shape, self.loc.shape[0]))  # global RNG
            x = self.loc + mx.exp(self.log_scale) * eps
            return x, self.log_prob(x)

    mx.random.seed(0)

    # Maximum likelihood on data
    x = 2.0 + 0.5 * mx.random.normal((5000, 2))
    model, losses = fit_to_data(
        DiagGaussian(2),
        x,
        loss_fn=maximum_likelihood_loss,
        learning_rate=1e-2,
        max_epochs=50,
    )
    print("MLE   loc:", model.loc, "scale:", mx.exp(model.log_scale))

    # Variational fit to an unnormalised target N(-1, 0.3^2)
    target = lambda z: (-0.5 * ((z + 1.0) / 0.3) ** 2).sum(-1)
    model, losses = fit_to_loss(
        DiagGaussian(2),
        loss_fn=make_elbo_loss(target, num_samples=128),
        steps=1000,
        learning_rate=1e-2,
    )
    print("ELBO  loc:", model.loc, "scale:", mx.exp(model.log_scale))
