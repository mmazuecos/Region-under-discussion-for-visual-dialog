import argparse

import csv

from utils import load_data, save_data, progressbar

from copy import deepcopy


def run(args):

    games = load_data(args.guesswhat_games)

    # get indices of samples in the mini test set
    games_new = []
    with open(args.csv_samples, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in progressbar(csv_reader):
            gid = int(row["Game ID"])
            pos = int(row["Position"])
            g = [deepcopy(g) for g in games if g["id"] == gid][0]
            try:
                g["qas"] = [g["qas"][pos], ]
                games_new.append(g)
            except:
                print(gid, pos, len(g["qas"]), g["qas"])

    save_data(games_new, args.output_file)
    print(f"{args.output_file} saved ({len(games_new)} games)")


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Add focus into GuessWhat?! annotation files",
        add_help=True,
        allow_abbrev=False
    )

    parser.add_argument(
        "guesswhat_games",
        help="GuessWhat?! games file (test set)",
        type=str
    )

    parser.add_argument(
        "csv_samples",
        help="mini test set samples csv",
        type=str,
    )

    parser.add_argument(
        "--output-file",
        help="output file",
        type=str,
        default="./output.jsonl.gz"
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    print("{}".format(vars(args)))
    run(args)
