from typing import Any, Tuple

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from h5py._hl.dataset import Dataset
from jaxtyping import Array, ArrayLike
from numpy.typing import NDArray

from .data import BasinData
from .models import AbstractModel


def _take_out_nans(y_true: NDArray, y_pred: NDArray) -> Tuple[NDArray, NDArray]:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return y_true[mask], y_pred[mask]


def get_kge(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_true, y_pred = _take_out_nans(y_true, y_pred)
    return 1 - np.sqrt(
        (np.corrcoef(y_true, y_pred)[0, 1] - 1) ** 2
        + (np.std(y_pred) / np.std(y_true) - 1) ** 2
        + (np.mean(y_pred) / np.mean(y_true) - 1) ** 2
    )


def get_nse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_true, y_pred = _take_out_nans(y_true, y_pred)
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


def get_pred_and_true(
    timeseries_dir: str,
    simulation_file_path: str,
    test_dates: Tuple,
    seq_length: int,
    basin_id: str,
) -> Tuple[NDArray, NDArray]:
    # pred
    with h5py.File(simulation_file_path, "r") as f:
        ds = f.get(basin_id)
        if isinstance(ds, Dataset):
            q_pred = ds[:]
        else:
            raise TypeError(f"Expected h5py._hl.dataset.Dataset, got {type(ds)}")

    # true
    basin_df = pd.DataFrame(pd.read_csv(timeseries_dir + f"/basin_{basin_id}.csv"))
    q_true = basin_df.streamflow.values
    start_date, end_date = pd.to_datetime(test_dates[0]), pd.to_datetime(test_dates[1])
    dates = pd.to_datetime(basin_df.date).values
    mask = (dates >= start_date) & (dates <= end_date)
    q_true = q_true[mask]
    q_true = q_true[seq_length - 1 :]

    return q_pred, q_true


def evaluate(
    model: AbstractModel,
    data: BasinData,
    xd_norms: Tuple[Array, Array],
    xs_norms: Tuple[Any, Any],
    y_norms: Tuple[Array, Array],
) -> Tuple[Array, Array]:
    # simulation
    y_pred = model.simulate(data, xd_norms, xs_norms, y_norms)
    y_true = data.y[:, model.seq_length - 1 :]

    # evaluation
    nses = []
    kges = []
    for i in range(y_true.shape[0]):
        nses.append(get_nse(y_true[i], y_pred[i]))
        kges.append(get_kge(y_true[i], y_pred[i]))

    return jnp.median(jnp.array(nses)), jnp.median(jnp.array(kges))


def show_timeseries(
    dates: NDArray,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    model_name: str,
) -> None:
    _, ax = plt.subplots(figsize=(8, 4))
    plt.plot(dates, y_true, label="Observation")
    plt.plot(dates, y_pred, label=model_name)
    ax.set_ylabel("Streamflow (m³/s)")
    plt.legend()
    plt.show()
