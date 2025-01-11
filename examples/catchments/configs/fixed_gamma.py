import ml_collections, optax, os
import jax.random as jrandom
import numpy as np

from jacare.models import FixedGamma

def get_config():
    config = ml_collections.ConfigDict()
    
    # dataset parameters
    config.target_name = "streamflow"
    config.timeseries_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../data/timeseries"
    )
    config.attributes_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../data/attributes"
    )
    config.train_ids = np.array([123, 456])
    config.val_ids = np.array([123, 456])
    config.test_ids = np.array([123, 456])
    config.ids_per_eval = 2
    config.mass_features_names = ["sro_sum", "ssro_sum"]
    config.additional_features_names = []
    config.area_name = "area"
    config.additional_attributes_names = []
    config.train_dates = (np.datetime64('1991-01-01'), np.datetime64('1992-12-31'))
    config.validation_dates = (np.datetime64('1991-01-01'), np.datetime64('1992-12-31'))
    config.test_dates = (np.datetime64('1991-01-01'), np.datetime64('1992-12-31'))
    
    # model parameters
    config.model_name = "fixed_gamma"
    config.shape = 1.5
    config.scale = 1.0
    config.seq_length = 60
    config.is_conserving_mass = True
    
    # model
    config.model = FixedGamma(
        shape=config.shape,
        scale=config.scale,
        seq_length=config.seq_length,
        is_conserving_mass=config.is_conserving_mass
    )
    
    # dummy training parameters just to save the model
    config.print_every = 1
    config.batch_size = 1
    config.learning_rate = 1e-1
    config.steps = 1
    config.save_every = 1
    config.max_save_to_keep = 1
    config.saving_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../checkpoints/" + config.model_name
    )
    config.optim = optax.adam(config.learning_rate)
    config.key = jrandom.PRNGKey(42)
    
    return config
