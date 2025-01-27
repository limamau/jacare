import h5py
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any, Tuple

from jacare.data import BasinData
from jacare.models import AbstractModel


def _take_out_nans(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return y_true[mask], y_pred[mask]


def get_kge(y_true, y_pred):
    y_true, y_pred = _take_out_nans(y_true, y_pred)
    return 1 - np.sqrt(
        (np.corrcoef(y_true, y_pred)[0, 1] - 1) ** 2 +
        (np.std(y_pred) / np.std(y_true) - 1) ** 2 +
        (np.mean(y_pred) / np.mean(y_true) - 1) ** 2
    )


def get_nse(y_true, y_pred):
    y_true, y_pred = _take_out_nans(y_true, y_pred)
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    

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


def evaluate(
    model: AbstractModel,
    data: BasinData,
    xd_norms: Tuple[jnp.ndarray, jnp.ndarray],
    xs_norms: Tuple[Any, Any],
    y_norms: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[float, float]:

    # simulation
    y_pred = model.simulate(data, xd_norms, xs_norms, y_norms)
    y_true = data.y[:,model.seq_length-1:]
    
    # evaluation
    nses = []
    kges = []
    for i in range(y_true.shape[0]):
        nses.append(get_nse(y_true[i], y_pred[i]))
        kges.append(get_kge(y_true[i], y_pred[i]))
    
    return jnp.median(jnp.array(nses)), jnp.median(jnp.array(kges))


def show_timeseries(dates, y_true, y_pred, model_name, title=None):
    _, ax = plt.subplots(figsize=(8,4))
    plt.plot(dates, y_true, label="Observation")
    plt.plot(dates, y_pred, label=model_name)
    ax.set_ylabel("Streamflow (m³/s)")
    plt.title(title)
    plt.legend()
    plt.show()
