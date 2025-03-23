import os

import numpy as np
import optax
from jax import random as jrandom

from jacare.models import MLPGamma


class Config:
    # dataset parameters
    target_name = "streamflow"
    timeseries_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/timeseries"
    )
    attributes_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../data/attributes"
    )
    train_ids = [1, 2]
    val_ids = [1, 2]
    test_ids = [1, 2]
    ids_per_eval = 2
    mass_features_names = ["sro_sum", "ssro_sum"]
    additional_features_names = []
    area_name = "area"
    additional_attributes_names = []
    train_dates = (np.datetime64("1991-01-01"), np.datetime64("1992-12-31"))
    validation_dates = (np.datetime64("1991-01-01"), np.datetime64("1992-12-31"))
    test_dates = (np.datetime64("1991-01-01"), np.datetime64("1992-12-31"))

    # model parameters
    model_name = "mlp_gamma"
    hidden_size = 32
    seed = 5678
    seq_length = 10
    multiplier = 5.0
    is_conserving_mass = True

    # model
    model = MLPGamma(
        attributes_size=len(additional_features_names),
        hidden_size=hidden_size,
        seq_length=seq_length,
        multiplier=multiplier,
        key=jrandom.PRNGKey(seed),
        is_conserving_mass=is_conserving_mass,
    )

    # training parameters
    print_every = 1  # 100
    batch_size = 1  # 128
    learning_rate = 3e-3
    steps = 11
    save_every = 100
    max_save_to_keep = 1
    saving_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../checkpoints/" + model_name,
    )
    optim = optax.adam(learning_rate)
    key = jrandom.PRNGKey(42)
