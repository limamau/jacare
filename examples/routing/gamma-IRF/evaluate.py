import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jacare.evaluation import get_pred_and_true
from jacare.routing import HillslopeChannelRouter

from configs import get_config


def get_pred(config, basin):
    # get config
    cfg = get_config(config)
    label = cfg.label
    timeseries_dir = cfg.timeseries_dir
    attributes_dir = cfg.attributes_dir
    routing_lvs_dir = cfg.routing_lvs_dir
    graph_file_path = cfg.graph_file_path
    simulation_file_path = cfg.simulation_file_path
    mass_features_names = cfg.mass_features_names
    additional_features_names = cfg.additional_features_names
    area_name = cfg.area_name
    distance_name = cfg.distance_name
    additional_attributes_names = cfg.additional_attributes_names
    target_name = cfg.target_name
    test_dates = cfg.test_dates
    hillslope_model = cfg.hillslope_model
    channel_model = cfg.channel_model
    method = cfg.method
    
    # dummy values for normalization
    hillslope_norms = {
        'xd_norms': (jnp.zeros((2,)), jnp.ones((2,))), # sro, ssro
        'xs_norms': (jnp.zeros((1,)), jnp.ones((1,))), # area
        'y_norms': (jnp.array([0.0]), jnp.array([1.0])), # streamflow
    }
    channel_norms = {
        'xd_norms': (jnp.zeros((3,)), jnp.ones((3,))), # sro, ssro, up_q
        'xs_norms': (jnp.zeros((2,)), jnp.ones((2,))), # area, distance
        'y_norms': (jnp.array([0.0]), jnp.array([1.0])), # streamflow
    }
    
    # define router
    router = HillslopeChannelRouter(
        timeseries_dir,
        attributes_dir,
        routing_lvs_dir,
        graph_file_path,
        simulation_file_path,
        mass_features_names,
        additional_features_names,
        area_name,
        distance_name,
        additional_attributes_names,
        target_name,
        test_dates,
        hillslope_model,
        channel_model,
        method,
    )
    
    # perform simulation
    router.simulate(*hillslope_norms.values(), *channel_norms.values())
    
    # get timeseries for the chosen basin
    q_pred, q_true = get_pred_and_true(
        timeseries_dir, 
        simulation_file_path,
        test_dates,
        hillslope_model.seq_length,
        basin,
    )
    
    return label, q_pred, q_true


def main(args):
    # plot for comparison
    _, ax = plt.subplots(figsize=(8,4))
    basin = args.basin
    
    # iterate through given configs
    flag = True
    for config in args.config:
        if flag:
            flag = False
            label, pred, q_true = get_pred(config, basin)
            dates = np.arange(pred.shape[-1])
            plt.plot(dates, q_true, label=f"{basin} - observed")
        
        else:
            label, pred, _ = get_pred(config, basin)
    
        plt.plot(dates, pred, label=f"{basin} - {label}")
    
    # finalize plot
    ax.set_ylabel("Streamflow (m³/s)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation with given configurations."
    )
    # choose configurations from the ones
    # inside configs/
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
    )
    # choose the basin ID to show from the ones
    # available in dataset
    parser.add_argument(
        "--basin",
        required=True,
    )
    args = parser.parse_args()
    
    main(args)
