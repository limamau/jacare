import os

import numpy as np

from jacare.models import FixedGamma, FixedIRF

DAY_TO_S = 86400.0


class Config:
    label = "parallel a)"

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
        "../../../../data/simulations/gamma-IRF/a.h5",
    )
    mass_features_names = ["sro_sum", "ssro_sum"]
    additional_features_names = []
    area_name = "area"
    distance_name = "distance"
    additional_attributes_names = []
    test_dates = (np.datetime64("1991-01-01"), np.datetime64("1991-12-31"))

    # hillslope model parameters
    hillslope_model_name = "gamma"
    shape = 1.5
    scale = 1.0
    seq_length = 120

    # hillslope model
    hillslope_model = FixedGamma(
        shape=shape,
        scale=scale,
        seq_length=seq_length,
    )

    # channel model parameters
    channel_model_name = "IRF"
    velocity = 1.0 * DAY_TO_S
    diffusivity = 800 * DAY_TO_S
    is_conserving_mass = True
    channel_model = FixedIRF(
        velocity=velocity,
        diffusivity=diffusivity,
        seq_length=seq_length,
        is_conserving_mass=is_conserving_mass,
    )

    # router
    method = "parallel"
