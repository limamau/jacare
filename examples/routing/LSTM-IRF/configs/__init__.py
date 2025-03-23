from argparse import Namespace

from . import config


def get_config(args: Namespace):
    if args.config == "config":
        return config.Config()
    else:
        raise ValueError(f"Unknown config: {args.config}")
