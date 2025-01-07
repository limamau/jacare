import json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DAY_TO_S = 86400
KM2_TO_M2 = 1000000


def generate_graph(script_dir):
    graph = {
        789: [123, 456],
        123: [],
        456: []
    }
    
    graph_file_path = os.path.join(script_dir, "graph.json")
    with open(graph_file_path, 'w') as f:
        json.dump(graph, f, indent=4)


def generate_routing_lvs(script_dir):
    routing_lv1 = [123, 456]
    routing_lv2 = [789]
    
    os.makedirs(os.path.join(script_dir, "routing_lvs"), exist_ok=True)
    
    with open(os.path.join(script_dir, "routing_lvs/routing_lv01.txt"), 'w') as f:
        f.writelines(f"{basin_id}\n" for basin_id in routing_lv1)
    
    with open(os.path.join(script_dir, "routing_lvs/routing_lv02.txt"), 'w') as f:
        f.writelines(f"{basin_id}\n" for basin_id in routing_lv2)
        

def generate_attributes(script_dir):
    AREA_CORRECTOR = 0.1
    area_123 = 60502.4 / 2 * AREA_CORRECTOR
    area_456 = 60502.4 * 2 * AREA_CORRECTOR
    area_789 = (area_123 + area_456) / 2 * AREA_CORRECTOR
    
    attributes_df = pd.DataFrame(
        index=pd.Index([123, 456, 789], name="basin"),
        data={
            "area": [area_123, area_456, area_789],
            "distance": [10000.0, 20000.0, 0.0], # dummy
        }
    )
    
    # save
    os.makedirs(os.path.join(script_dir, "attributes"), exist_ok=True)
    attributes_df.to_csv(os.path.join(script_dir, "attributes/attributes.csv"))


def generate_basins(script_dir, orig_file_path):
    orig_df = pd.read_csv(orig_file_path)
    attributes_names = ["sro_sum", "ssro_sum", "t2m_mean", "streamflow"]
    orig = orig_df[attributes_names].values
    
    # 123
    basin_123_df = pd.DataFrame(
        index=orig_df.date[0:3650],
        data={
            "sro_sum": orig[0:3650, 0],
            "ssro_sum": orig[0:3650, 1],
            "t2m_mean": orig[0:3650, 2],
            "streamflow": orig[0:3650, 3] / 2,
        }
    )
    
    # 456
    basin_456_df = pd.DataFrame(
        index=orig_df.date[0:3650],
        data={
            "sro_sum": orig[0:3650, 0],
            "ssro_sum": orig[0:3650, 1],
            "t2m_mean": orig[0:3650, 2],
            "streamflow": orig[0:3650, 3] * 2,
        }
    )
    
    # time-shifted and aggregated streamflows
    streamflow_123_shifted = basin_123_df.streamflow.shift(10, fill_value=0)
    streamflow_456_shifted = basin_456_df.streamflow.shift(20, fill_value=0)
    orig_streamflow = orig[0:3650, 3]
    streamflow_789 = streamflow_123_shifted + streamflow_456_shifted + orig_streamflow
    
    # 789
    basin_789_df = pd.DataFrame(
        index=basin_123_df.index,
        data={
            "sro_sum": basin_123_df.sro_sum + basin_456_df.sro_sum,
            "ssro_sum": basin_123_df.ssro_sum + basin_456_df.ssro_sum,
            "t2m_mean": (basin_123_df.t2m_mean + basin_456_df.t2m_mean),
            "streamflow": streamflow_789,
        }
    )
    
    # save basins
    os.makedirs(os.path.join(script_dir, "timeseries"), exist_ok=True)
    basin_123_df.to_csv(os.path.join(script_dir, "timeseries/basin_123.csv"))
    basin_456_df.to_csv(os.path.join(script_dir, "timeseries/basin_456.csv"))
    basin_789_df.to_csv(os.path.join(script_dir, "timeseries/basin_789.csv"))
    
    return basin_123_df, basin_456_df, basin_789_df
    
def plot_streamflows(basin_123_df, basin_456_df, basin_789_df):
    plt.figure(figsize=(10, 6))
    plt.plot(basin_123_df.index, basin_123_df.streamflow, label="Basin 123")
    plt.plot(basin_456_df.index, basin_456_df.streamflow, label="Basin 456")
    plt.plot(basin_789_df.index, basin_789_df.streamflow, label="Basin 789", linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Streamflow")
    plt.title("Streamflows for Basins 123, 456, and 789")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # generate graph and routing levels
    generate_graph(script_dir)
    generate_routing_lvs(script_dir)
    
    # timeseries
    orig_file_path = os.path.join(
        script_dir,
        "basin_orig.csv",
    )
    basin_123_df, basin_456_df, basin_789_df = generate_basins(script_dir, orig_file_path)
    
    # attributes
    generate_attributes(script_dir)
    
    # quick check
    plot_streamflows(basin_123_df, basin_456_df, basin_789_df)

if __name__ == "__main__":
    main()
