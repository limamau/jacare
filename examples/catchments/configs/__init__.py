from . import fixed_gamma, lstm, mlp_gamma


def get_config(cfg):
    if cfg == "fixed_gamma":
        return fixed_gamma.get_config()
    elif cfg == "mlp_gamma":
        return mlp_gamma.get_config()
    elif cfg == "lstm":
        return lstm.get_config()
    else:
        raise ValueError(f"Unknown config: {cfg}")
