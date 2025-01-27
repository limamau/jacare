import argparse, h5py
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from jacare.checkpointing import Checkpointer
from jacare.routing import HillslopeChannelRouter

from configs import get_config


def get_pred_and_true(
    timeseries_dir, 
    simulation_file_path,
    test_dates,
    seq_length,
    basin_id,
):
    # pred
    with h5py.File(simulation_file_path, "r") as f:
        q_pred = f[basin_id][:]
        
    # true
    basin_df = pd.read_csv(timeseries_dir + f"/basin_{basin_id}.csv")
    q_true = basin_df["streamflow"].values
    start_date, end_date = pd.to_datetime(test_dates[0]), pd.to_datetime(test_dates[1])
    dates = pd.to_datetime(basin_df["date"]).values
    mask = (dates >= start_date) & (dates <= end_date)
    q_true = q_true[mask]
    q_true = q_true[seq_length-1:]
    
    return q_pred, q_true


def main(args):
    # get config
    cfg = get_config(args.config)
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
    checkpoint_path = cfg.checkpoint_path
    channel_model = cfg.channel_model
    
    # restore hillslope model from checkpoint in examples/catchments
    hillslope_model, hillslope_norms = Checkpointer.restore_latest(
        hillslope_model,
        checkpoint_path,
        len(mass_features_names + additional_features_names), # mass + additional features
        1 + len(additional_attributes_names), # area + additional attributes
    )
    
    # dummy values for normalization of channel model
    channel_norms_dict = {
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
    _, ax = plt.subplots(figsize=(8,4))
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
