import ml_collections, os
import numpy as np

from jacare.models import FixedGamma, FixedIRF


DAY_TO_S = 86400.0

def get_config():
    config = ml_collections.ConfigDict()
    
    # dataset parameters
    config.target_name = "streamflow"
    config.timeseries_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../data/timeseries"
    )
    config.attributes_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../data/attributes"
    )
    config.routing_lvs_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../data/routing_lvs"
    )
    config.graph_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../data/graph.json"
    )
    config.simulation_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../data/simulations/gamma-IRF/b.h5"
    )
    config.mass_features_names = ["sro_sum", "ssro_sum"]
    config.additional_features_names = []
    config.area_name = "area"
    config.distance_name = "distance"
    config.additional_attributes_names = []
    config.test_dates = (np.datetime64('1991-01-01'), np.datetime64('1991-12-31'))
    
    #hillslope model parameters
    config.hillslope_model_name = "gamma"
    config.shape = 1.5
    config.scale = 1.0
    config.seq_length = 120
    
    # hillslope model
    config.hillslope_model = FixedGamma(
        shape=config.shape,
        scale=config.scale,
        seq_length=config.seq_length,
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
