import jax, os
import jax.numpy as jnp
import orbax.checkpoint as ocp
from typing import Tuple

from jacare.models import AbstractModel

class Checkpointer():
    mngr: ocp.CheckpointManager
    norm_tree: dict
    
    def __init__(
        self,
        saving_path: str,
        max_save_to_keep: int,
        save_every: int,
        xd_norms: Tuple[jnp.ndarray, jnp.ndarray],
        xs_norms: Tuple[jnp.ndarray, jnp.ndarray],
        y_norms: Tuple[jnp.ndarray, jnp.ndarray],
    ):
        self.saving_path = saving_path
        path = ocp.test_utils.erase_and_create_empty(saving_path)
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_save_to_keep,
            save_interval_steps=save_every,
        )
        os.makedirs(path, exist_ok=True)
        self.mngr = ocp.CheckpointManager(
            path, options=options, item_names=('model', 'norms'),
        )
        self.norm_tree = {
            'xd_norms': xd_norms,
            'xs_norms': xs_norms,
            'y_norms': y_norms,
        }
    
    @staticmethod
    def get_abstract_args(
            model: AbstractModel,
            num_dynamic_features,
            num_static_features,
        ):
        
        abstract_model = jax.tree_util.tree_map(
            ocp.utils.to_shape_dtype_struct, model,
        )
        
        norm_tree = {
                'xd_norms': (
                    jnp.zeros((num_dynamic_features)),
                    jnp.zeros((num_dynamic_features)),
                ),
                'xs_norms': (
                    jnp.zeros((num_static_features)),
                    jnp.zeros((num_static_features)),
                ),
                'y_norms': (jnp.array([0.0]), jnp.array([0.0])),
            }
        abstract_norm = jax.tree_util.tree_map(
            ocp.utils.to_shape_dtype_struct, norm_tree,
        )
            
        return abstract_model, abstract_norm

    @classmethod
    def restore_latest(
        self,
        model: AbstractModel,
        saving_path: str,
        num_dynamic_features: int,
        num_static_features: int,
    ):
        # get abstract args
        abstract_model, abstract_norm = self.get_abstract_args(
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
                model=ocp.args.StandardRestore(abstract_model),
                norms=ocp.args.StandardRestore(abstract_norm),
            ),
        )
        model, norm_dict = restored.model, restored.norms
        
        return model, norm_dict.values()

    def save(self, model, step):
        self.mngr.save(
            step,
            args=ocp.args.Composite(
                model=ocp.args.StandardSave(model),
                norms=ocp.args.StandardSave(self.norm_tree),
            ),
        )
