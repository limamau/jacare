from argparse import Namespace

from . import fixed_gamma, lstm, mlp_gamma


def get_config(args: Namespace):
    if args.config == "fixed_gamma":
        return fixed_gamma.Config()
    elif args.config == "mlp_gamma":
        return mlp_gamma.Config()
    elif args.config == "lstm":
        return lstm.Config()
    else:
        raise ValueError(f"Unknown config: {args.config}")
