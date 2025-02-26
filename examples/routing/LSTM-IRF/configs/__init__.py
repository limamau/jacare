from . import config


def get_config(cfg):
    if cfg == "config":
        return config.get_config()
    else:
        raise ValueError(f"Unknown config: {cfg}")
