from . import a_iterative
from . import b_iterative
from . import a_parallel
from . import b_parallel

def get_config(config):
    if config == "a_iterative":
        return a_iterative.get_config()
    elif config == "b_iterative":
        return b_iterative.get_config()
    elif config == "a_parallel":
        return a_parallel.get_config()
    elif config == "b_parallel":
        return b_parallel.get_config()
