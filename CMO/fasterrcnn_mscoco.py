import os

import argparse

import numpy as np

from PIL import Image

import torch

from fasterrcnn import FasterRCNNFeatures

from utils import progressbar, draw_bounding_boxes, xyxy2xywh

import glob


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
        num_objects=args.num_objects,
        with_attributes=args.with_attributes,
        device=device)

    # ------------------------------------------------------------------------

    imlist = glob.glob(
        os.path.join(args.coco_root, "**", "*.jpg"), recursive=True
    )

    # imlist = [os.path.split(im)[1] for im in imlist]  # strip path

    coco_root = os.path.join(os.path.commonpath(imlist), "")
    imlist = [f.replace(coco_root, "") for f in imlist]

    n_train = len([im for im in imlist if "train2014" in im])
    n_val = len([im for im in imlist if "val2014" in im])
    n_test = len([im for im in imlist if "test2014" in im])

    print(f"{n_train} train, {n_val} val, {n_test} test")

    # odict = {}

    for imfile in progressbar(imlist):
        # coco_set = imfile.split("_")[1]
        # impath = os.path.join(os.path.abspath(args.coco_root), coco_set, imfile)
        impath = os.path.join(coco_root, imfile)
        im = Image.open(impath).convert("RGB")

        instances, features = detector(im)
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

        # imkey = os.path.splitext(imfile)[0]
        # odict[imkey] = {
        odict = {
            "image_height": instances.image_size[0],
            "image_width": instances.image_size[1],
            "scores": scores,
            "classes": classes,
            "boxes": xyxy2xywh(boxes) if len(boxes) > 0 else boxes,
            "features": features
        }

        if args.with_attributes:
            odict["attributes"] = [
                detector.attributes[i]
                for i in instances.attr_classes.tolist()
            ]

        # import matplotlib.pyplot as plt
        # topk = 10
        # print(imfile, odict[gid]["boxes"][:topk], classes)
        # im = draw_bounding_boxes(im, odict["boxes"][:topk], labels=classes[:topk], fmt="xywh", color=(223, 223, 0))
        # print(im.size, len(boxes))
        # plt.imshow(im)
        # plt.show()
        # continue

        npy_fname = os.path.join(
            os.path.abspath(args.output_path),
            f"mscoco_num-objects_{args.num_objects}" + (
                "_with_attributes" if args.with_attributes else ""
            ),
            os.path.splitext(imfile)[0] + ".npy"
        )
        os.makedirs(os.path.dirname(npy_fname), exist_ok=True)
        np.save(npy_fname, odict)


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute FasterRCNN features on COCO images",
        add_help=True,
        allow_abbrev=False
    )

    # datasets
    parser.add_argument(
        "--coco-root",
        help="COCO dataset path",
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
        "--num-objects",
        help="get <num-objects> objects with the highest scores",
        type=int,
        default=36
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
