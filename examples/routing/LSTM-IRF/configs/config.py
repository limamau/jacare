import os

import jax.random as jrandom
import ml_collections
import numpy as np

from jacare.models import LSTM, FixedIRF

DAY_TO_S = 86400.0


# here no training parameters are defined
# as we are only savinig fixed values to evaluate the model
def get_config():
    config = ml_collections.ConfigDict()

    # dataset parameters
    config.target_name = "streamflow"
    config.timeseries_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../../data/timeseries"
    )
    config.attributes_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../../data/attributes"
    )
    config.routing_lvs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../../data/routing_lvs"
    )
    config.graph_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../../data/graph.json"
    )
    config.simulation_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../data/simulations/lstm-IRF/sim.h5",
    )
    config.mass_features_names = ["sro_sum", "ssro_sum"]
    config.additional_features_names = []
    config.area_name = "area"
    config.distance_name = "distance"
    config.additional_attributes_names = []
    config.test_dates = (np.datetime64("1991-01-01"), np.datetime64("1991-12-31"))

    # hillslope model parameters
    config.model_name = "lstm"
    config.hidden_size = 16
    config.seed = 5678
    config.seq_length = 60

    # hillslope model
    config.hillslope_model = LSTM(
        in_size=len(
            config.mass_features_names
            + config.additional_features_names
            + config.additional_attributes_names
        )
        + 1,  # for area
        hidden_size=config.hidden_size,
        seq_length=config.seq_length,
        key=jrandom.PRNGKey(config.seed),
    )
    config.checkpoint_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../catchments/checkpoints/" + config.model_name,
    )

    # channel model parameters
    config.channel_model_name = "IRF"
    config.velocity = 0.001 * DAY_TO_S
    config.diffusivity = 8000 * DAY_TO_S
    config.channel_model = FixedIRF(
        velocity=config.velocity,
        diffusivity=config.diffusivity,
        seq_length=config.seq_length,
    )

    return config
