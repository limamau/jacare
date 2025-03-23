import os

import jax.random as jrandom
import numpy as np

from jacare.models import LSTM, FixedIRF

DAY_TO_S = 86400.0


# here no training parameters are defined
# as we are only savinig fixed values to evaluate the model
class Config:
    # dataset parameters
    target_name = "streamflow"
    timeseries_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../../data/timeseries"
    )
    attributes_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../../data/attributes"
    )
    routing_lvs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../../data/routing_lvs"
    )
    graph_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../../data/graph.json"
    )
    simulation_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../data/simulations/lstm-IRF/sim.h5",
    )
    mass_features_names = ["sro_sum", "ssro_sum"]
    additional_features_names = []
    area_name = "area"
    distance_name = "distance"
    additional_attributes_names = []
    test_dates = (np.datetime64("1991-01-01"), np.datetime64("1991-12-31"))

    # hillslope model parameters
    model_name = "lstm"
    hidden_size = 16
    seed = 5678
    seq_length = 60

    # hillslope model
    hillslope_model = LSTM(
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
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../catchments/checkpoints/" + model_name,
    )

    # channel model parameters
    channel_model_name = "IRF"
    velocity = 0.001 * DAY_TO_S
    diffusivity = 8000 * DAY_TO_S
    channel_model = FixedIRF(
        velocity=velocity,
        diffusivity=diffusivity,
        seq_length=seq_length,
    )
