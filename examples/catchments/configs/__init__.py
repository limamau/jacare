from . import fixed_gamma
from . import lstm
from . import mlp_gamma

def get_config(args):
    if args.config == "fixed_gamma":
        return fixed_gamma.get_config()
    elif args.config == "mlp_gamma":
        return mlp_gamma.get_config()
    elif args.config == "lstm":
        return lstm.get_config()
