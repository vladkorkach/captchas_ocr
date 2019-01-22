import argparse
from network.utils import train_labels_metrics_reader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="epoch to investigate",
        required=True
    )
    parser.add_argument(
        "-s",
        "--show",
        type=bool,
        help="",
        default=False
    )

    args = vars(parser.parse_args())
    epoch = args['epochs']
    show = args["show"]

    train_labels_metrics_reader(epoch=epoch, show_incorrect=show)
