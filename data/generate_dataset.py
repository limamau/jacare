import json
import os

import matplotlib.pyplot as plt
import pandas as pd

DAY_TO_S = 86400
KM2_TO_M2 = 1000000


def generate_graph(script_dir: str):
    graph = {
        1: [],
        2: [],
        3: [1, 2],
        4: [3],
    }

    graph_file_path = os.path.join(script_dir, "graph.json")
    with open(graph_file_path, "w") as f:
        json.dump(graph, f, indent=4)


def generate_routing_lvs(script_dir: str):
    routing_lv1 = [1, 2]
    routing_lv2 = [3]
    routing_lv3 = [4]

    os.makedirs(os.path.join(script_dir, "routing_lvs"), exist_ok=True)

    with open(os.path.join(script_dir, "routing_lvs/routing_lv01.txt"), "w") as f:
        f.writelines(f"{basin_id}\n" for basin_id in routing_lv1)

    with open(os.path.join(script_dir, "routing_lvs/routing_lv02.txt"), "w") as f:
        f.writelines(f"{basin_id}\n" for basin_id in routing_lv2)

    with open(os.path.join(script_dir, "routing_lvs/routing_lv03.txt"), "w") as f:
        f.writelines(f"{basin_id}\n" for basin_id in routing_lv3)


def generate_attributes(script_dir: str):
    AREA_CORRECTOR = 0.1
    area_1 = 60502.4 / 2 * AREA_CORRECTOR
    area_2 = 60502.4 * 2 * AREA_CORRECTOR
    area_3 = (area_1 + area_2) / 2 * AREA_CORRECTOR
    area_4 = area_3

    attributes_df = pd.DataFrame(
        index=pd.Index([1, 2, 3, 4], name="basin"),
        data={
            "area": [area_1, area_2, area_3, area_4],
            "distance": [10000.0, 20000.0, 5000.0, 0.0],
        },
    )

    # save
    os.makedirs(os.path.join(script_dir, "attributes"), exist_ok=True)
    attributes_df.to_csv(os.path.join(script_dir, "attributes/attributes.csv"))


def generate_basins(script_dir: str, orig_file_path: str):
    orig_df = pd.DataFrame(pd.read_csv(orig_file_path))
    attributes_names = ["sro_sum", "ssro_sum", "t2m_mean", "streamflow"]
    orig = orig_df.loc[:, attributes_names].values

    # 1
    basin_1_df = pd.DataFrame(
        index=orig_df.date[0:3650],
        data={
            "sro_sum": orig[0:3650, 0],
            "ssro_sum": orig[0:3650, 1],
            "t2m_mean": orig[0:3650, 2],
            "streamflow": orig[0:3650, 3] / 2,
        },
    )

    # 2
    basin_2_df = pd.DataFrame(
        index=orig_df.date[0:3650],
        data={
            "sro_sum": orig[0:3650, 0],
            "ssro_sum": orig[0:3650, 1],
            "t2m_mean": orig[0:3650, 2],
            "streamflow": orig[0:3650, 3] * 2,
        },
    )

    # time-shifted and aggregated streamflows
    streamflow_1_shifted = basin_1_df.streamflow.shift(10, fill_value=0)
    streamflow_2_shifted = basin_2_df.streamflow.shift(20, fill_value=0)
    orig_streamflow = orig[0:3650, 3]
    streamflow_3 = streamflow_1_shifted + streamflow_2_shifted + orig_streamflow
    streamflow_4 = streamflow_3.shift(5, fill_value=0)

    # 3
    basin_3_df = pd.DataFrame(
        index=basin_1_df.index,
        data={
            "sro_sum": basin_1_df.sro_sum + basin_2_df.sro_sum,
            "ssro_sum": basin_1_df.ssro_sum + basin_2_df.ssro_sum,
            "t2m_mean": (basin_1_df.t2m_mean + basin_2_df.t2m_mean),
            "streamflow": streamflow_3,
        },
    )

    # 4
    basin_4_df = pd.DataFrame(
        index=basin_1_df.index,
        data={
            "sro_sum": basin_3_df.sro_sum,
            "ssro_sum": basin_3_df.ssro_sum,
            "t2m_mean": basin_3_df.t2m_mean,
            "streamflow": streamflow_4,
        },
    )

    # save basins
    os.makedirs(os.path.join(script_dir, "timeseries"), exist_ok=True)
    basin_1_df.to_csv(os.path.join(script_dir, "timeseries/basin_1.csv"))
    basin_2_df.to_csv(os.path.join(script_dir, "timeseries/basin_2.csv"))
    basin_3_df.to_csv(os.path.join(script_dir, "timeseries/basin_3.csv"))
    basin_4_df.to_csv(os.path.join(script_dir, "timeseries/basin_4.csv"))

    return basin_1_df, basin_2_df, basin_3_df, basin_4_df


def plot_streamflows(
    basin_1_df: pd.DataFrame,
    basin_2_df: pd.DataFrame,
    basin_3_df: pd.DataFrame,
    basin_4_df: pd.DataFrame,
):
    plt.figure(figsize=(10, 6))
    plt.plot(basin_1_df.index, basin_1_df.streamflow, label="Basin 1")
    plt.plot(basin_2_df.index, basin_2_df.streamflow, label="Basin 2")
    plt.plot(basin_3_df.index, basin_3_df.streamflow, label="Basin 3")
    plt.plot(basin_4_df.index, basin_4_df.streamflow, label="Basin 4")
    plt.xlabel("Date")
    plt.ylabel("Streamflow")
    plt.title("Streamflows for Basins 1, 2, 3 and 4")
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
    basin_1_df, basin_2_df, basin_3_df, basin_4_df = generate_basins(
        script_dir, orig_file_path
    )

    # attributes
    generate_attributes(script_dir)

    # quick check
    plot_streamflows(basin_1_df, basin_2_df, basin_3_df, basin_4_df)


if __name__ == "__main__":
    main()
