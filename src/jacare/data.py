import os
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pandas as pd
from typing import Any, Tuple, List


class BasinData:
    def __init__(
        self, 
        xd: np.ndarray, 
        xs: np.ndarray, 
        y: np.ndarray, 
        basin_ids: List[int]
    ):
        self.xd = xd
        self.xs = xs
        self.y = y
        self.basin_ids = basin_ids

    @staticmethod
    def filter_dates(
        x: np.ndarray, 
        y: np.ndarray,
        dates: np.ndarray,
        date_bounds: Tuple[str, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        start_date, end_date = pd.to_datetime(date_bounds[0]), pd.to_datetime(date_bounds[1])
        mask = (dates >= start_date) & (dates <= end_date)
        return x[mask], y[mask]

    @classmethod
    def from_files(
        cls,
        timeseries_dir: str,
        attributes_dir: str,
        basin_ids_list: List[int],
        mass_features_names: List[str],
        additional_features_names: List[str],
        area_name: str,
        additional_attributes_names: List[str],
        target_name: str,
        bound_dates: Tuple[str, str],
    ):
        all_xd, all_y, all_xs = [], [], []

        for basin_id in basin_ids_list:
            # load dynamic data
            df = pd.read_csv(os.path.join(timeseries_dir, f"basin_{basin_id}.csv"))
            dates = pd.to_datetime(df["date"]).values
            xd = df[mass_features_names + additional_features_names].values
            y = df[target_name].values

            # filter dates
            xd, y = cls.filter_dates(xd, y, dates, bound_dates)

            # load static data
            attributes_df = pd.read_csv(os.path.join(attributes_dir, "attributes.csv"))
            xs = attributes_df[attributes_df.basin == basin_id][
                [area_name] + additional_attributes_names
            ].values

            # append to lists
            all_xd.append(xd)
            all_y.append(y)
            all_xs.append(xs)

        # convert lists to arrays
        xd = np.stack(all_xd)
        y = np.stack(all_y)
        xs = np.vstack(all_xs) 

        return cls(xd, xs, y, basin_ids_list)    
    
    def get_single_basin(
        self, 
        idx: int
    ):
        return self.xd[idx], self.xs[idx], self.y[idx]
    
    def get_random_subset(
        self, 
        key: jrandom.PRNGKey,
        size: int,
    ) -> Tuple[jnp.ndarray, Any, jnp.ndarray]:

        if size > self.basin_ids.shape[0]:
            raise ValueError("Subset size cannot be larger than the total number of basins.")

        # generate random indices
        indices = jrandom.choice(
            key, 
            self.y.shape[0],
            shape=(size,),
            replace=False
        )

        # select corresponding data
        xd_subset = jnp.take(self.xd, indices, axis=0)
        y_subset = jnp.take(self.y, indices, axis=0)

        if self.xs is not None:
            xs_subset = jnp.take(self.xs, indices, axis=0)
        else:
            xs_subset = None

        # create a BasinData object with an empty list of basin ids
        # as this is not used in the code that calls this function (until now)
        return BasinData(xd_subset, xs_subset, y_subset, [])
    
    def get_norms(self):
        xd_norms = (
            jnp.mean(self.xd, axis=(0,1)),
            jnp.std(self.xd, axis=(0,1)),
        )
        xs_norms = (
            jnp.mean(self.xs, axis=0), 
            jnp.std(self.xs, axis=0),
        )
        y_norms = (
            jnp.array([jnp.mean(self.y)]), 
            jnp.array([jnp.std(self.y)]),
        )
        return xd_norms, xs_norms, y_norms

    def normalize(
        self,
        xd_norms: Tuple[np.ndarray, np.ndarray],
        xs_norms: Tuple[np.ndarray, np.ndarray],
        y_norms: np.ndarray,
        epsilon=1e-18
    ):
        self.xd = (self.xd - xd_norms[0]) / (xd_norms[1] + epsilon)
        self.xs = (self.xs - xs_norms[0]) / (xs_norms[1] + epsilon)
        if y_norms is not None:
            self.y = (self.y - y_norms[0]) / (y_norms[1] + epsilon)
