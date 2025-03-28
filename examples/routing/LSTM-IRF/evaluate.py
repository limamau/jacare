import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from configs import get_config

from jacare.checkpointing import Checkpointer
from jacare.evaluation import get_pred_and_true
from jacare.routing import HillslopeChannelRouter


def main(args):
    # get config
    config = get_config(args)
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
    checkpoint_path = config.checkpoint_path
    channel_model = config.channel_model

    # restore hillslope model from checkpoint in examples/catchments
    hillslope_model, hillslope_norms = Checkpointer.restore_latest(
        hillslope_model,
        checkpoint_path,
        len(
            mass_features_names + additional_features_names
        ),  # mass + additional features
        1 + len(additional_attributes_names),  # area + additional attributes
    )

    # dummy values for normalization of channel model
    channel_norms_dict = {
        "xd_norms": (jnp.zeros((3,)), jnp.ones((3,))),  # sro, ssro, up_q
        "xs_norms": (jnp.zeros((2,)), jnp.ones((2,))),  # area, distance
        "y_norms": (jnp.array([0.0]), jnp.array([1.0])),  # streamflow
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
    )

    # perform simulation
    router.simulate(*hillslope_norms, *channel_norms_dict.values())

    # get simulations from saved file
    seq_length = hillslope_model.seq_length
    basin_id = "4"
    q_pred, q_true = get_pred_and_true(
        timeseries_dir,
        simulation_file_path,
        test_dates,
        seq_length,
        basin_id,
    )

    # quick check
    _, ax = plt.subplots(figsize=(8, 4))
    dates = np.arange(q_pred.shape[-1])
    plt.plot(dates, q_true, label="Observations")
    plt.plot(dates, q_pred, label="LSTM-IRF")
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
        required=True,
    )
    args = parser.parse_args()

    main(args)
