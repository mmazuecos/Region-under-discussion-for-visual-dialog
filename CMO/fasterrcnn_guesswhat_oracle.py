import os

import argparse

import numpy as np

from PIL import Image

import torch

from fasterrcnn import FasterRCNNFeatures

from utils import progressbar, load_data, xyxy2xywh, xywh2xyxy, draw_bounding_boxes


def run(args):
    # ------------------------------------------------------------------------
    # preamble

    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_threads)
    num_workers = 2 * args.num_threads

    # ------------------------------------------------------------------------
    # Detection model

    detector = FasterRCNNFeatures(
        args.py_bottom_up_attention_path,
        with_attributes=args.with_attributes,
        device=device)

    # ------------------------------------------------------------------------
    # GuessWhat! dataset

    jsonfile = os.path.join(
        args.guesswhat_root, "data", f"guesswhat.{args.set}.jsonl.gz"
    )
    games = load_data(jsonfile)

    # ------------------------------------------------------------------------

    odict = {}

    for g in progressbar(games):
        imfile = g["image"]["file_name"]
        coco_set = "train2014" if "train2014" in imfile else "val2014"
        if coco_set in args.coco_root:
            imfile = os.path.join(args.coco_root, imfile)
        else:
            imfile = os.path.join(args.coco_root, coco_set, imfile)

        npy_fname = os.path.join(args.output_path, f"{coco_set}", g["image"]["file_name"])
        if os.path.exists(npy_fname):
            continue

        im = Image.open(imfile).convert("RGB")

        target_box = np.atleast_2d([
            obj["bbox"]
            for obj in g["objects"]
            if obj["id"] == g["object_id"]
        ])

        instances, features = detector.predict(im, xywh2xyxy(target_box))
        instances = instances.to("cpu")
        features = features.to("cpu")

        scores = instances.scores.squeeze().numpy().astype(np.float32)

        classes = [
            detector.classes[i]
            for i in instances.pred_classes.tolist()
        ]

        boxes = np.array([
            b.squeeze().tolist() for b in instances.pred_boxes
        ]).astype(np.float32)

        features = features.numpy().astype(np.float32)

        gid = str(g["id"])

        odict[gid] = {
            "image_height": instances.image_size[0],
            "image_width": instances.image_size[1],
            "scores": scores,
            "classes": classes,
            "boxes": xyxy2xywh(boxes) if len(boxes) > 0 else boxes,
            "features": features
        }

        if args.with_attributes:
            odict[gid]["attributes"] = [
                detector.attributes[i]
                for i in instances.attr_classes.tolist()
            ]

        # import matplotlib.pyplot as plt
        # topk = 10
        # print(imfile, odict[gid]["boxes"][:topk], classes)
        # im = draw_bounding_boxes(im, odict[gid]["boxes"][:topk], labels=classes[:topk], fmt="xywh", color=(223, 223, 0))
        # print(im.size, len(boxes))
        # plt.imshow(im)
        # plt.show()
        # continue

    npy_fname = f"guesswhat_{args.set}_oracle"
    if args.with_attributes:
        npy_fname += "_with_attributes"
    npy_fname = os.path.join(
        os.path.abspath(args.output_path), npy_fname + ".npy"
    )
    os.makedirs(os.path.dirname(npy_fname), exist_ok=True)
    np.save(npy_fname, odict)


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute FasterRCNN features for the GuessWhat?! dataset",
        add_help=True,
        allow_abbrev=False
    )

    # datasets
    parser.add_argument(
        "--guesswhat-root",
        help="GuessWhat! dataset path",
        type=str,
        default="./guesswhat"
    )

    parser.add_argument(
        "--set",
        help="GuessWhat! set",
        type=str,
        default="valid"
    )

    parser.add_argument(
        "--coco-root",
        help="COCO root path",
        type=str,
        default="./COCO"
    )

    parser.add_argument(
        "--py-bottom-up-attention-path",
        help="py-bottom-up-attention path",
        type=str,
        default="./py-bottom-up-attention"
    )

    parser.add_argument(
        "--with-attributes",
        help="use attributes model",
        action="store_true",
    )

    parser.add_argument(
        "--output-path",
        help="output path",
        type=str,
        default="./fasterrcnn/"
    )

    # running options
    parser.add_argument(
        "--num-threads",
        help="torch num threads",
        type=int,
        default=8
    )

    parser.add_argument(
        "--seed",
        help="random seed",
        type=int,
        default=1234
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    print("{}".format(vars(args)))
    run(args)
