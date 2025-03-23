import os

import jax.random as jrandom
import numpy as np
import optax

from jacare.models import FixedGamma


class Config:
    # dataset parameters
    target_name = "streamflow"
    timeseries_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../data/timeseries"
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
    model_name = "fixed_gamma"
    shape = 1.5
    scale = 1.0
    seq_length = 60
    is_conserving_mass = True

    # model
    model = FixedGamma(
        shape=shape,
        scale=scale,
        seq_length=seq_length,
        is_conserving_mass=is_conserving_mass,
    )

    # dummy training parameters just to save the model
    print_every = 1
    batch_size = 1
    learning_rate = 1e-1
    steps = 1
    save_every = 1
    max_save_to_keep = 1
    saving_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../checkpoints/" + model_name,
    )
    optim = optax.adam(learning_rate)
    key = jrandom.PRNGKey(42)
