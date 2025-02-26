import os
import shutil
from typing import Tuple

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jaxtyping import Array

from .models import AbstractModel


class Checkpointer:
    mngr: ocp.CheckpointManager
    norm_tree: dict

    def __init__(
        self,
        saving_path: str,
        max_save_to_keep: int,
        save_every: int,
        xd_norms: Tuple[Array, Array],
        xs_norms: Tuple[Array, Array],
        y_norms: Tuple[Array, Array],
        erase: bool = False,
    ):
        self.saving_path = saving_path
        if erase:
            if os.path.exists(saving_path):
                shutil.rmtree(saving_path)
        os.makedirs(saving_path, exist_ok=True)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_save_to_keep,
            save_interval_steps=save_every,
        )
        self.mngr = ocp.CheckpointManager(
            saving_path,
            options=options,
            item_names=("model", "norms"),
        )
        self.norm_tree = {
            "xd_norms": xd_norms,
            "xs_norms": xs_norms,
            "y_norms": y_norms,
        }

    @staticmethod
    def get_abstract_args(
        model: AbstractModel,
        num_dynamic_features: int,
        num_static_features: int,
    ) -> Tuple[
        AbstractModel,
        Tuple[Tuple[Array, Array], Tuple[Array, Array], Tuple[Array, Array]],
    ]:
        abstract_model = jax.tree_util.tree_map(
            ocp.utils.to_shape_dtype_struct,
            model,
        )

        norm_tree = {
            "xd_norms": (
                jnp.zeros((num_dynamic_features)),
                jnp.zeros((num_dynamic_features)),
            ),
            "xs_norms": (
                jnp.zeros((num_static_features)),
                jnp.zeros((num_static_features)),
            ),
            "y_norms": (jnp.array([0.0]), jnp.array([0.0])),
        }
        abstract_norm = jax.tree_util.tree_map(
            ocp.utils.to_shape_dtype_struct,
            norm_tree,
        )

        return abstract_model, abstract_norm

    @classmethod
    def restore_latest(
        cls,
        model: AbstractModel,
        saving_path: str,
        num_dynamic_features: int,
        num_static_features: int,
    ) -> Tuple[AbstractModel, Tuple[Tuple[Array, Array], ...]]:
        # get abstract args
        abstract_model, abstract_norm = cls.get_abstract_args(
            model=model,
            num_dynamic_features=num_dynamic_features,
            num_static_features=num_static_features,
        )

        # restore manager
        mngr = ocp.CheckpointManager(
            saving_path,
        )

        # restores
        restored = mngr.restore(
            mngr.latest_step(),
            args=ocp.args.Composite(
                **{
                    "model": ocp.args.StandardRestore(abstract_model),  # pyright: ignore
                    "norms": ocp.args.StandardRestore(abstract_norm),  # pyright: ignore
                }
            ),
        )
        model, norm_dict = restored["model"], restored["norms"]

        return model, norm_dict.values()

    def save(self, model: AbstractModel, step: int) -> None:
        self.mngr.save(
            step,
            args=ocp.args.Composite(
                **{
                    "model": ocp.args.StandardSave(model),  # pyright: ignore
                    "norms": ocp.args.StandardSave(self.norm_tree),  # pyright: ignore
                }
            ),
        )
