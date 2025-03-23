from . import a_iterative, a_parallel, b_iterative, b_parallel


def get_config(config: str):
    if config == "a_iterative":
        return a_iterative.Config()
    elif config == "b_iterative":
        return b_iterative.Config()
    elif config == "a_parallel":
        return a_parallel.Config()
    elif config == "b_parallel":
        return b_parallel.Config()
    else:
        raise ValueError(f"Unknown config: {config}")
