import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jacare.evaluation import get_pred_and_true
from jacare.routing import HillslopeChannelRouter

from configs import get_config


def get_pred(config):
    # get config
    config = get_config(config)
    basin = args.basin
    timeseries_dir = config.timeseries_dir
    attributes_dir = config.attributes_dir
    routing_lvs_dir = config.routing_lvs_dir
    graph_file_path = config.graph_file_path
    simulation_file_path = config.simulation_file_path
    mass_features_names = config.mass_features_names
    additional_features_names = config.additional_features_names
    area_name = config.area_name
    distance_name = config.distance_name
    additional_attributes_names = config.additional_attributes_names
    target_name = config.target_name
    test_dates = config.test_dates
    hillslope_model = config.hillslope_model
    channel_model = config.channel_model
    method = config.method

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

    return q_pred, q_true


def main(args):
    # plot for comparison
    _, ax = plt.subplots(figsize=(8,4))
    basin = args.basin

    # iterate through given configs
    flag = True
    for config in args.config:
        if flag:
            flag = False
            pred, q_true = get_pred(config)
            dates = np.arange(pred.shape[-1])
            plt.plot(dates, q_true, label=f"basin {basin} - observed")

        else:
            pred, _ = get_pred(config)

        plt.plot(dates, pred, label=f"basin {basin} - {config}") # pyright: ignore

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
