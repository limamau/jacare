import os
import numpy as np
import optax
from jax import random as jrandom

from jacare.models import LSTM


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
    model_name = "lstm"
    hidden_size = 16
    seed = 5678
    seq_length = 60

    # model
    model = LSTM(
        in_size=len(
            mass_features_names
            + additional_features_names
            + additional_attributes_names
        )
        + 1,  # for area
        hidden_size=hidden_size,
        seq_length=seq_length,
        key=jrandom.PRNGKey(seed),
    )

    # training parameters
    print_every = 100
    batch_size = 128
    learning_rate = 3e-3
    steps = 1001
    save_every = 100
    max_save_to_keep = 1
    saving_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../checkpoints/" + model_name,
    )
    optim = optax.adam(learning_rate)
    key = jrandom.PRNGKey(42)
