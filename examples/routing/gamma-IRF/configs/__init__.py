from . import a_iterative, a_parallel, b_iterative, b_parallel


def get_config(cfg):
    if cfg == "a_iterative":
        return a_iterative.get_config()
    elif cfg == "b_iterative":
        return b_iterative.get_config()
    elif cfg == "a_parallel":
        return a_parallel.get_config()
    elif cfg == "b_parallel":
        return b_parallel.get_config()
    else:
        raise ValueError(f"Unknown config: {cfg}")
