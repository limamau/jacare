from . import fixed_gamma
from . import lstm
from . import mlp_gamma

def get_config(cfg):
    if cfg == "fixed_gamma":
        return fixed_gamma.get_config()
    elif cfg == "mlp_gamma":
        return mlp_gamma.get_config()
    elif cfg == "lstm":
        return lstm.get_config()
    else:
        raise ValueError(f"Unknown config: {cfg}")
