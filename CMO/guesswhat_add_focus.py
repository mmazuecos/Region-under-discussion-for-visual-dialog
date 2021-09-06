import argparse

import json

from utils import load_data, save_data, progressbar


def run(args):

    games = load_data(args.guesswhat_games)

    focus = json.load(open(args.focus, "r"))

    # as a list while keeping the order given by the str(int) keys
    focus = [focus[str(i)] for i in range(len(focus))]

    for i, g in progressbar(enumerate(games), total=len(games)):
        # if not success, set empty focus
        if g["status"] != "success":
            for j in range(len(g["qas"])):
                games[i]["qas"][j]["focus"] = None
            continue

        # look up for indices that match the current game id
        idxs = [
            k
            for k, data in enumerate(focus)
            if int(data["game_id"]) == g["id"]
        ]

        # they must be consistent in length with the current dialog
        if len(idxs) != len(g["qas"]):
            raise RuntimeError(f"inconsistent data for game {g['id']} (status={g['status']})")

        target = [
            obj["bbox"]
            for obj in g["objects"]
            if obj["id"] == g["object_id"]
        ]

        focus_bbox = None
        for j, k in enumerate(idxs):

            # # update focus bbox if present
            # if focus[k]["focus_bbox"] is not None:
            #     focus_bbox = focus[k]["focus_bbox"]

            # update focus bbox if present
            if focus[k]["focus_bbox"] is not None:
                # if set to the target, it means the history contradicts itself
                # and it must therefore be reset
                if np.allclose(focus[k]["focus_bbox"], target, atol=1e-2):
                    focus_bbox = (0, 0, g["image"]["width"]-1, g["image"]["height"]-1)
                else:
                    focus_bbox = focus[k]["focus_bbox"]

            if focus_bbox is None:
                games[i]["qas"][j]["focus"] = None
            else:
                x1, y1, x2, y2 = focus_bbox
                assert ((x2 > x1) and (y2 > y1))  # xyxy format
                games[i]["qas"][j]["focus"] = (x1, y1, x2-x1, y2-y1)  # xywh format

        # idxs are countiguous. Remove these to speed up next iter
        del focus[min(idxs):max(idxs)+1]

    save_data(games, args.output_file)
    print(f"{args.output_file} saved")


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Add focus into GuessWhat?! annotation files",
        add_help=True,
        allow_abbrev=False
    )

    parser.add_argument(
        "guesswhat_games",
        help="GuessWhat?! games file",
        type=str
    )

    parser.add_argument(
        "focus",
        help="json file with focus data (for the same set)",
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
