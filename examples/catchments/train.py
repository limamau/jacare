import argparse

from jacare.data import BasinData
from jacare.training import train_routing_level

from configs import get_config

    
def main(args):
    # get config
    cfg = get_config(args.config)
    timeseries_dir = cfg.timeseries_dir
    attributes_dir = cfg.attributes_dir
    train_ids = cfg.train_ids
    val_ids = cfg.val_ids
    ids_per_eval = cfg.ids_per_eval
    mass_features_names = cfg.mass_features_names
    additional_features_names = cfg.additional_features_names
    area_name = cfg.area_name
    additional_attributes_names = cfg.additional_attributes_names
    target_name = cfg.target_name
    train_dates = cfg.train_dates
    validation_dates = cfg.validation_dates
    print_every = cfg.print_every
    batch_size = cfg.batch_size
    steps = cfg.steps
    save_every = cfg.save_every
    max_save_to_keep = cfg.max_save_to_keep
    saving_path = cfg.saving_path
    key = cfg.key
    model = cfg.model
    optim = cfg.optim
    
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
