import ml_collections, optax, os
from jax import random as jrandom
import numpy as np

from jacare.models import MLPGamma

def get_config():
    config = ml_collections.ConfigDict()
    
    # dataset parameters
    config.target_name = "streamflow"
    config.timeseries_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../data/timeseries"
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
    config.model_name = "mlp_gamma"
    config.hidden_size = 32
    config.seed = 5678
    config.seq_length = 10
    config.multiplier = 5.0
    
    # model
    config.model = MLPGamma(
        attributes_size=len(config.additional_features_names),
        hidden_size=config.hidden_size,
        seq_length=config.seq_length,
        multiplier=config.multiplier,
        key=jrandom.PRNGKey(config.seed)
    )
    
    # training parameters
    config.print_every = 1 # 100
    config.batch_size = 1 # 128
    config.learning_rate = 3e-3
    config.steps = 11
    config.save_every = 100
    config.max_save_to_keep = 1
    config.saving_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../checkpoints/" + config.model_name
    )
    config.optim = optax.adam(config.learning_rate)
    config.key = jrandom.PRNGKey(42)
    
    return config
