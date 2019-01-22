import argparse
from network.train import train_dispather
from config import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-tb",
        "--tensorboard",
        type=bool,
        help="use tensorboard for metrics",
        default=True,
    )

    args = vars(parser.parse_args())

    tensorboard = args["tensorboard"]

    train_dispather()
