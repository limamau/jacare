import h5py, os, json
import jax.numpy as jnp
import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import List

from jacare.data import BasinData
from jacare.models import AbstractModel


class AbstractRouter():
    timeseries_dir: str
    attributes_dir: str
    routing_lvs_basins: list[list[int]]
    num_routing_lvs: int
    graph: dict
    simulation_file_path: str
    num_routing_lvs: int
    
    def simulate(self, *norms):
        os.makedirs(os.path.dirname(self.simulation_file_path), exist_ok=True)
        for routing_lv in range(1, self.num_routing_lvs+1):
            self.simulate_routing_lv(routing_lv, *norms)
    
    @abstractmethod    
    def simulate_routing_lv(self):
        pass
    
    # TODO: add simulate_batch method
    # to simulate a batch inside a routing level
    # and define the simulate_routing_lv method
    # iniside this class
    
    # TODO: add option to simulate a sub-ensemble
    # of basins on this routing level


class HillslopeChannelRouter(AbstractRouter):
    hillslope_model: AbstractModel
    channel_model: AbstractModel
    
    def __init__(
        self,
        timeseries_dir: str,
        attributes_dir: str,
        routing_lvs_dir: str,
        graph_file_path: str,
        simulation_file_path: str,
        mass_features_names: List[str],
        additional_features_names: List[str],
        area_name: str,
        distance_name: str,
        additional_attributes_names: List[str],
        target_name: str,
        bound_dates: str,
        hillslope_module: AbstractModel,
        channel_module: AbstractModel,
        method: str = 'iterative',
        # TODO: add optional variable "target" to alleviate
        # training and include all basins ('all') or only gauged basins
        # ('gauged') and their upstreams in the graph? Another option could 
        # be to do that automatically during trainig, but using only gauged
        # basins can be useful even for simulations with the goal of 
        # only evaluating the model
    ):
        self.timeseries_dir = timeseries_dir
        self.attributes_dir = attributes_dir
        self.num_routing_lvs = len(os.listdir(routing_lvs_dir))
        self.routing_lvs_basins = read_routing_lvs(
            routing_lvs_dir, self.num_routing_lvs
        )
        self.graph = read_graph(graph_file_path)
        
        self.method = method
        # method == iterative will go iteratively through all
        # routing levels where each basin is connected to a list
        # containing direct upstreams only
        if self.method == 'iterative':
            pass
        # method == parallel will transform all routing levels > 1
        # into one (routing level 2) where each basin is connected 
        # to a list containing all upstreams
        elif self.method == 'parallel':
            self.graph = aggregate_descendents(self.graph)
            self.num_routing_lvs = 1
            self.routing_lvs_basins = aggregate_routing_lvs(self.routing_lvs_basins)
        else:
            raise ValueError("Only methods 'iterative' and 'parallel' are valid.")
        
        self.graph = integrate_distances_on_graph(self.graph, attributes_dir, distance_name)
        self.simulation_file_path = simulation_file_path
        self.mass_features_names = mass_features_names
        self.additional_features_names = additional_features_names
        self.area_name = area_name
        self.additional_attributes_names = additional_attributes_names
        self.target_name = target_name
        self.bound_dates = bound_dates
        self.hillslope_model = hillslope_module
        self.channel_model = channel_module

    @staticmethod
    def merge_channel_data(data, up_dict, seq_length):
        # array of summed streamflow from upstreams
        xd_upq = np.zeros((data.xd.shape[0], data.xd.shape[1]-seq_length+1))
        
        # array of mean distance from upstreams
        xs_upl = np.zeros((data.xs.shape[0], 1))
        
        for i, basin_id in enumerate(data.basin_ids):
            xd_upq[i] = np.sum([value for value, _ in up_dict[basin_id]], axis=0)
            xs_upl[i] = np.mean([value for _, value in up_dict[basin_id]], axis=0)
        
        # fill missing values with zero to match shape
        xd_upq = jnp.concatenate(
            [jnp.zeros((data.xd.shape[0], seq_length-1)), xd_upq],
            axis=1,
        )
        # concat summed upstreamflow to xd as new feature
        xd_upq = jnp.expand_dims(xd_upq, axis=2)
        data.xd = jnp.concatenate((data.xd, xd_upq), axis=2)
        
        # concat mean distance from upstreams in xs
        data.xs = jnp.concatenate((data.xs, xs_upl), axis=1)
    
    def simulate_routing_lv(
        self, routing_lv,
        hillslope_xd_norms, hillslope_xs_norms, hillslope_y_norms,
        channel_xd_norms, channel_xs_norms, channel_y_norms,
    ):
        basin_ids = self.routing_lvs_basins[routing_lv-1]
        
        # hillslope contribution
        data = BasinData.from_files(
            timeseries_dir=self.timeseries_dir,
            attributes_dir=self.attributes_dir,
            basin_ids_list=basin_ids,
            mass_features_names=self.mass_features_names,
            additional_features_names=self.additional_features_names,
            area_name=self.area_name,
            additional_attributes_names=self.additional_attributes_names,
            target_name=self.target_name,
            bound_dates=self.bound_dates,
        )
        qHS = self.hillslope_model.simulate(
            data,
            hillslope_xd_norms, hillslope_xs_norms, hillslope_y_norms,
        )
        
        # channel contribution
        if routing_lv == 1:
            qC = 0
        else:
            up_dict = get_up_dict(
                self.graph,
                self.simulation_file_path,
                basin_ids,
            )
            
            # TODO: what if up_dict was directly used?
            # one idea is to define the models between
            # "hillslope" models, which take a fixed size ndarray as argument
            # and "channel" models, which take that and a dictionary.
            # Should this dictionary be part of BasinData?
            # Can this be an argument of this class?
            self.merge_channel_data(data, up_dict, self.hillslope_model.seq_length)
            
            qC = self.channel_model.simulate(
                data,
                channel_xd_norms, channel_xs_norms, channel_y_norms,
            )
        
        # aggregation
        q = qHS + qC
        save_simulation(self.simulation_file_path, basin_ids, q)


# list of auxiliary functions to be used by all Routers
# would it be better to put them as static methods inside the AbstractRiverRouter class?

def aggregate_descendents(
    graph: dict,
):
    new_graph = {}
    
    def get_all_upstreams(basin):
        direct_upstreams = graph[basin]
        if not direct_upstreams:
            return []
        else:
            all_upstreams = []
            for upstream in direct_upstreams:
                all_upstreams += [upstream] + get_all_upstreams(upstream)
            return all_upstreams
    
    for basin in graph.keys():
        new_graph[basin] = get_all_upstreams(basin)
        
    del graph
    return new_graph


def aggregate_routing_lvs(routing_lvs_basins):
    new_routing_lvs_basins = [[], []]
    flag = True
    for routing_lv_basins in routing_lvs_basins:
        if flag:
            new_routing_lvs_basins[0] = routing_lv_basins
            flag = False
        else:
            new_routing_lvs_basins[1] += routing_lv_basins
    del routing_lvs_basins
    return new_routing_lvs_basins
                

def get_up_dict(
    graph: dict,
    simulation_file_path: str,
    basin_ids: List[int],
):
    up_dict = {}
    
    with h5py.File(simulation_file_path, 'r') as f:
        for basin_id in basin_ids:
            items = graph.get(basin_id, [])
            
            flag = True
            for up_id, up_l in items:
                up_q = f[str(up_id)][:]
                if flag:
                    up_dict[basin_id] = [(up_q, up_l)]
                    flag = False
                else: 
                    up_dict[basin_id].append((up_q, up_l))
            
    return up_dict


def integrate_distances_on_graph(
    graph: dict,
    attributes_dir: str,
    distance_name: str,
):
    attributes_df = pd.read_csv(
        os.path.join(attributes_dir, 'attributes.csv'),
        index_col='basin',
    )
    
    new_graph = {}
    
    for basin_id, upstream_ids in graph.items():
        if not upstream_ids:
            new_graph[basin_id] = []
        
        flag = True
        for upstream_id in upstream_ids:
            l0 = attributes_df.loc[basin_id, distance_name]
            l = attributes_df.loc[upstream_id, distance_name] - l0
            if flag:
                new_graph[basin_id] = [(upstream_id, l)]
                flag = False
            else:
                new_graph[basin_id].append((upstream_id, l))
            
    return new_graph


def read_graph(file_path: str):
    with open(file_path, 'r') as json_file:
        graph = json.load(json_file)
    
    # keys and values are converted to int
    # TODO: save and read as string is better?
    return {int(k): [int(v) for v in vals] for k, vals in graph.items()}

def read_routing_lvs(routing_lvs_dir: str, num_routing_lvs: int):
    def read_routing_lv(routing_lvs_dir: str, routing_lv: int):
        file_path = os.path.join(
            routing_lvs_dir, f"routing_lv{routing_lv:02d}.txt"
        )
        with open(file_path, 'r') as file:
            basin_ids = [int(line.strip()) for line in file]
        return basin_ids
    
    routing_lvs = []
    for routing_lv in range(1, num_routing_lvs+1):
        routing_lvs.append(read_routing_lv(routing_lvs_dir, routing_lv))
    
    return routing_lvs



def save_simulation(
    simulation_file_path: str, 
    basin_ids: List[int], 
    q: jnp.ndarray,
):
    with h5py.File(simulation_file_path, 'a') as f:
        for i, basin_id in enumerate(basin_ids):
            dataset_name = str(basin_id)
            # remove existing dataset to avoid conflicts
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(dataset_name, data=q[i])
