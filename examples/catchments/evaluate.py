import argparse
import time
import numpy as np

from jacare.checkpointing import Checkpointer
from jacare.data import BasinData
from jacare.evaluation import get_kge, get_nse, show_timeseries

from configs import get_config


def main(args):
    # get config
    config = get_config(args)
    model_name = config.model_name
    timeseries_dir = config.timeseries_dir
    attributes_dir = config.attributes_dir
    mass_features_names = config.mass_features_names
    additional_features_names = config.additional_features_names
    area_name = config.area_name
    additional_attributes_names = config.additional_attributes_names
    test_ids = config.test_ids
    target_name = config.target_name
    test_dates = config.test_dates
    saving_path = config.saving_path
    model = config.model

    # get dataset
    test_data = BasinData.from_files(
        timeseries_dir,
        attributes_dir,
        test_ids,
        mass_features_names,
        additional_features_names,
        area_name,
        additional_attributes_names,
        target_name,
        test_dates,
    )

    # restore
    model, norms = Checkpointer.restore_latest(
        model,
        saving_path,
        # mass + additional features
        len(mass_features_names) + len(additional_features_names),
        1 + len(additional_attributes_names),  # area + additional attributes
    )

    # simulation
    start_time = time.time()
    y_pred = model.simulate(test_data, *norms)
    end_time = time.time()
    print("Simulation time:", end_time - start_time)
    y_true = test_data.y[:, model.seq_length-1:]

    # metrics
    nse = get_nse(y_true, y_pred)
    kge = get_kge(y_true, y_pred)
    print("median_nse:", nse)
    print("median_kge:", kge)

    # quick check
    show_timeseries(
        dates=np.arange(y_pred.shape[-1]),
        y_true=y_true[0, :],
        y_pred=y_pred[0, :],
        model_name=model_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation with given configuration."
    )
    parser.add_argument(
        "--config",
        required=True,
    )
    args = parser.parse_args()

    main(args)
