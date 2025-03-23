import argparse

from jacare.data import BasinData
from jacare.training import train_routing_level

from configs import get_config


def main(args):
    # get config
    config = get_config(args)
    timeseries_dir = config.timeseries_dir
    attributes_dir = config.attributes_dir
    train_ids = config.train_ids
    val_ids = config.val_ids
    ids_per_eval = config.ids_per_eval
    mass_features_names = config.mass_features_names
    additional_features_names = config.additional_features_names
    area_name = config.area_name
    additional_attributes_names = config.additional_attributes_names
    target_name = config.target_name
    train_dates = config.train_dates
    validation_dates = config.validation_dates
    print_every = config.print_every
    batch_size = config.batch_size
    steps = config.steps
    save_every = config.save_every
    max_save_to_keep = config.max_save_to_keep
    saving_path = config.saving_path
    key = config.key
    model = config.model
    optim = config.optim

    # load data manually
    train_data = BasinData.from_files(
        timeseries_dir,
        attributes_dir,
        train_ids,
        mass_features_names,
        additional_features_names,
        area_name,
        additional_attributes_names,
        target_name,
        train_dates,
    )
    val_data = BasinData.from_files(
        timeseries_dir,
        attributes_dir,
        val_ids,
        mass_features_names,
        additional_features_names,
        area_name,
        additional_attributes_names,
        target_name,
        validation_dates,
    )

    # train the model with the
    # only routing level there is
    train_routing_level(
        model,
        optim,
        train_data,
        val_data,
        ids_per_eval,
        batch_size,
        steps,
        print_every,
        save_every,
        max_save_to_keep,
        saving_path,
        key,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training with given configuration."
    )
    parser.add_argument(
        "--config",
        required=True,
    )
    args = parser.parse_args()

    main(args)
