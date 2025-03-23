from typing import Iterator, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jaxtyping import Array, Key
from optax import GradientTransformation

from .checkpointing import Checkpointer
from .data import BasinData
from .evaluation import evaluate
from .models import AbstractModel


# auxiliary functions #
@eqx.filter_value_and_grad
def compute_loss(model: AbstractModel, *args) -> Array:
    *model_args, y = args
    pred_y = jax.vmap(model)(*model_args)
    # limamau: pass loss function as an argument
    mse = jnp.mean((y - pred_y) ** 2)
    return mse


def dataloader(
    arrays: Tuple[Array, ...], batch_size: int
) -> Iterator[Tuple[Array, ...]]:
    dataset_size = arrays[0].shape[0]
    indices = np.arange(dataset_size)
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


# training function #
def train_routing_level(
    model: AbstractModel,
    optim: GradientTransformation,
    train_data: BasinData,
    val_data: BasinData,
    ids_per_eval: int,
    batch_size: int,
    steps: int,
    print_every: int,
    save_every: int,
    max_save_to_keep: int,
    saving_path: str,
    key: Key,
) -> None:
    # preprocessing of training data
    norms = model.get_norms(train_data)
    train_data.normalize(*norms)
    args = model.serialize(train_data)
    iter_data = dataloader(args, batch_size)

    # initializations
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    ckpter = Checkpointer(
        saving_path,
        max_save_to_keep,
        save_every,
        *norms,
    )

    # forward pass
    @eqx.filter_jit
    def make_step(model, opt_state, *args):
        loss, grads = compute_loss(model, *args)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    # training loop
    losses = np.zeros((print_every,))
    for step, (args) in zip(range(steps), iter_data):
        loss, model, opt_state = make_step(model, opt_state, *args)
        losses[step % print_every] = loss

        # logging
        if step % print_every == 0 or step == steps - 1:
            key, skey = jrandom.split(key)
            subset_data = val_data.get_random_subset(skey, ids_per_eval)
            scores = evaluate(model, subset_data, *norms)
            step_msg = f"step {step},"
            loss_msg = f" train_loss={np.mean(losses):.4f},"
            nse_msg = f" val_nse={scores[0]:.2f},"
            kge_msg = f" val_kge={scores[1]:.2f}"
            print(step_msg + loss_msg + nse_msg + kge_msg, flush=True)

        # checkpointing
        ckpter.save(model, step)
    ckpter.mngr.wait_until_finished()


# limamau: def train_router():
# train hillslope model and save it
# then iteratively simulate and train each routing_lv > 1
# save after each complete route
# this function also defines two paths for saving each model and norms
# one for hillslope and one for channel routing
# (so the router needs to be HillslopeChannelRouter)
