import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jacare.routing import HillslopeChannelRouter

from configs.a import get_config as get_a
from configs.b import get_config as get_b

# TODO: write this function in in an utils.py file
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
    basin_123 = pd.read_csv(timeseries_dir + f"/basin_{basin_id}.csv")
    q_true = basin_123["streamflow"].values
    start_date, end_date = pd.to_datetime(test_dates[0]), pd.to_datetime(test_dates[1])
    dates = pd.to_datetime(basin_123["date"]).values
    mask = (dates >= start_date) & (dates <= end_date)
    q_true = q_true[mask]
    q_true = q_true[seq_length-1:]
    
    return q_pred, q_true


def get_preds(get_cfg):
    # get config
    cfg = get_cfg()
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
    )
    
    # perform simulation
    router.simulate(*hillslope_norms.values(), *channel_norms.values())
    
    # show timeseries for each of the basins
    q_pred_123, q_true = get_pred_and_true(
        timeseries_dir, 
        simulation_file_path,
        test_dates,
        hillslope_model.seq_length,
        "123",
    )
    
    q_pred_456, q_true = get_pred_and_true(
        timeseries_dir, 
        simulation_file_path,
        test_dates,
        hillslope_model.seq_length,
        "456",
    )
    
    q_pred_789, q_true = get_pred_and_true(
        timeseries_dir, 
        simulation_file_path,
        test_dates,
        hillslope_model.seq_length,
        "789",
    )
    
    return q_pred_123, q_pred_456, q_pred_789


def main():
    # experiment a)
    a_pred_123, a_pred_456, a_pred_789 = get_preds(get_cfg=get_a)
    
    # experiment b)
    b_pred_123, b_pred_456, b_pred_789 = get_preds(get_cfg=get_b)
    
    # quick comparison
    _, ax = plt.subplots(figsize=(8,4))
    dates = np.arange(a_pred_789.shape[-1])
    plt.plot(dates, a_pred_789, label="789 - a")
    plt.plot(dates, b_pred_789, label="789 - b")
    ax.set_ylabel("Streamflow (m³/s)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
