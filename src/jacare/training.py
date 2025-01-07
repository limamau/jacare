import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from typing import Callable

from jacare.checkpointing import Checkpointer
from jacare.data import BasinData
from jacare.evaluation import evaluate
from jacare.models import AbstractModel


@eqx.filter_value_and_grad
def compute_loss(model, *args):
    *model_args, y = args
    pred_y = jax.vmap(model)(*model_args)
    # TODO: pass loss function as an argument
    mse = jnp.mean((y - pred_y) ** 2)
    return mse


def dataloader(arrays, batch_size):
    dataset_size = arrays[0].shape[0]
    indices = np.arange(dataset_size)
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(
                array[batch_perm] for array in arrays
            )
            start = end
            end = start + batch_size


def train_routing_level(
    mod: AbstractModel,
    optim: Callable,
    train_data: BasinData,
    val_data: BasinData,
    ids_per_eval: int,
    batch_size: int,
    steps: int,
    print_every: int,
    save_every: int,
    max_save_to_keep: int,
    saving_path: str,
    key: jax.random.PRNGKey,
):
    # preprocessing of training data
    norms = mod.get_norms(train_data)
    train_data.normalize(*norms)
    args = mod.serialize(train_data)
    iter_data = dataloader((args), batch_size)

    # initializations
    opt_state = optim.init(mod)
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
        loss, mod, opt_state = make_step(mod, opt_state, *args)
        losses[step % print_every] = loss

        # logging
        if step % print_every == 0 or step == steps - 1:
            key, skey = jrandom.split(key)
            subset_data = val_data.get_random_subset(skey, ids_per_eval)
            scores = evaluate(mod, subset_data, *norms)
            step_msg = f"step {step},"
            loss_msg = f" train_loss={np.mean(losses):.4f},"
            nse_msg = f" val_nse={scores[0]:.2f},"
            kge_msg = f" val_kge={scores[1]:.2f}"
            print(step_msg + loss_msg + nse_msg + kge_msg)

        # checkpointing
        ckpter.save(mod, step)
    ckpter.mngr.wait_until_finished()
    

# TODO: def train_router():
# train hillslope model and save it
# then iteratively simulate and train each routing_lv > 1
# save after each complete route
# this function also defines two paths for saving each model and norms
# one for hillslope and one for channel routing
# (so the router needs to be HillslopeChannelRouter)
