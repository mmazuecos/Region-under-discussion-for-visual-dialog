import os

import argparse

from utils import timestamp


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self,
                 description="GuessWhat?! Oracle task",
                 with_model_params=True,
                 with_data_params=True,
                 with_learning_params=True,
                 with_dialog_history_params=False):
        super(ArgumentParser, self).__init__(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description=description,
            add_help=True,
            allow_abbrev=False
        )

        if with_data_params:
            self.add_data_parameters()

        if with_model_params:
            self.add_model_parameters()

        if with_learning_params:
            self.add_learning_parameters()

        if with_dialog_history_params:
            self.add_dialog_history_parameters()

        self.add_runtime_args()

    @staticmethod
    def args_to_path(args):
        path = os.path.join(os.path.abspath(args.cache), f"{timestamp()}")

        if hasattr(args, "max_length"):
            path += "_" + "_".join([
                f"max-length_{args.max_length}",
            ])

        if hasattr(args, "marker"):
            path += "_" + "_".join([
                f"marker_{args.marker}",
            ])

        if hasattr(args, "learning_rate"):
            path += "_" + "_".join([
                f"learning-rate_{args.learning_rate}",
                f"batch-size_{args.batch_size}",
                f"epochs_{args.epochs}",
                f"weight-decay_{args.weight_decay}",
                f"max-grad-norm_{args.max_grad_norm}"
            ])

        # if hasattr(args, "num_turns"):
        #     path += "_" + "_".join([
        #         f"num-turns_{args.num_turns}",
        #     ])

        if hasattr(args, "focus_mode"):
            path += "_" + "_".join([
                f"focus-mode_{args.focus_mode}",
            ])

        if args.suffix is not None:
            path += "_" + f"{args.suffix}"

        return path

    @staticmethod
    def path_to_args(path):
        path = os.path.abspath(path)
        if path.endswith(".pt") or os.path.isfile(path):
            path = os.path.dirname(path)
        path = path.rstrip(os.path.sep)
        path = os.path.split(path)[1].split("_")

        def parse_key_value(key, type_):
            # key: argument name
            # type_: expected type
            key_ = key.replace("-", "_")
            if key not in path:  # key not in path
                return (key_, None)
            elif type_ is bool:  # boolean args
                return (key_, True)
            else:  # key-value pairs
                return (key_, type_(path[path.index(key) + 1]))

        entries = [
            ("max-length", int),
            ("marker", str),
            ("learning-rate", float),
            ("batch-size", int),
            ("epochs", int),
            ("weight-decay", float),
            ("max-grad-norm", float),
            #("num-turns", int),
            ("focus-mode", str),
        ]
        dargs = dict(parse_key_value(k, t) for k, t in entries)

        return argparse.Namespace(**dargs)

    def add_data_parameters(self):
        group = self.add_argument_group("data parameters")

        group.add_argument(
            "--coco-data-root",
            help="features root path",
            type=str,
            default="./fasterrcnn/mscoco_num-objects_36"
        )

        group.add_argument(
            "--oracle-targets-root",
            help="oracle targets root path",
            type=str,
            default="./fasterrcnn"
        )

        group.add_argument(
            "--guesswhat-root",
            help="GuessWhat?! dataset path",
            type=str,
            default="./guesswhat"
        )

        group.add_argument(
            "--max-length",
            help="max sequence length",
            type=int,
            default=32
        )

    def add_model_parameters(self):
        group = self.add_argument_group("model parameters")

        group.add_argument(
            "--marker",
            help="target marker",
            type=str,
            default="affine",
            choices=("none", "affine", "offset")
        )

    def add_learning_parameters(self):
        group = self.add_argument_group("learning parameters")

        group.add_argument(
            "--learning-rate",
            help="learning rate",
            type=float,
            default=0.00001
        )

        group.add_argument(
            "--batch-size",
            help="batch size",
            type=int,
            default=32
        )

        group.add_argument(
            "--epochs",
            help="training epochs",
            type=int,
            default=5
        )

        group.add_argument(
            "--weight-decay",
            help="L2 regularization",
            type=float,
            default=0.0
        )

        group.add_argument(
            "--max-grad-norm",
            help="gradient clipping max norm",
            type=float,
            default=0.0
        )

        group.add_argument(
            "--checkpoint",
            help="resume from checkpoint",
            type=str
        )

    def add_dialog_history_parameters(self):
        group = self.add_argument_group("dialog history")

        # group.add_argument(
        #     "--num-turns",
        #     help="number of training turns",
        #     type=int,
        #     default=3
        # )

        group.add_argument(
            "--focus-mode",
            help="number of training turns",
            type=str,
            choices=("none", "relative", "restriction", "mixed", "zeros", "random"),
            default="none"
        )

    def add_runtime_args(self):
        group = self.add_argument_group("runtime arguments")

        group.add_argument(
            "--num-threads",
            help="torch num threads",
            type=int
        )

        group.add_argument(
            "--num-workers",
            help="dataloader num workers",
            type=int
        )

        group.add_argument(
            "--seed",
            help="random seed",
            type=int,
            default=1234
        )

        group.add_argument(
            "--suffix",
            help="path suffix",
            type=str
        )

        group.add_argument(
            "--cache",
            help="cache path",
            type=str,
            default="./cache"
        )
