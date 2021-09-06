import os

import argparse

import numpy as np

import torch

from utils import progressbar, cprint

from datasets import COCODataProvider, COCODataProviderSingleFile, GuessWhatOracle

import models as m

import transformers

from guesswhat_qtype_classifier import QTypeClassifier
#from qclassify import qclass as QTypeClassifier

from tabulate import tabulate

from parser import ArgumentParser


NATOK = "N/A"
#NATOK = "<NA>"


@torch.no_grad()
def eval_ep_qtype(loader, model):
    device = next(model.parameters()).device

    model.eval()

    total_hits = 0

    total_count = 0

    qtype_clf = QTypeClassifier()

    qtype_attrs = list(qtype_clf.qtype.keys()) + [NATOK,]
    #qtype_attrs = ["<color>", "<shape>", "<action>", "<size>", "<texture>", "<spatial>", "<object>", "<super-category>", NATOK]

    qtype_hits = {k: 0 for k in qtype_attrs}
    qtype_hits_with_focus = qtype_hits.copy()
    qtype_hits_without_focus = qtype_hits.copy()

    qtype_counts = {k: 0 for k in qtype_attrs}
    qtype_counts_with_focus = qtype_counts.copy()
    qtype_counts_without_focus = qtype_counts.copy()

    # We assume there is no shuffling. This keeps an index to the begining of the batch.
    idx = 0

    for visdata, langdata, targets in progressbar(loader, total=len(loader)):

        for k, v in visdata.items():
            visdata[k] = v.to(device)

        for k, v in langdata.items():
            langdata[k] = v.to(device)

        targets = targets.squeeze(-1).to(device)

        logits = model(visdata, langdata)

        preds = logits.detach().argmax(1)

        hits = (preds == targets).int().cpu().tolist()

        batch_size = targets.numel()

        for i in range(batch_size):
            q = loader.dataset.samples[idx + i]["question"]
            qtypes = qtype_clf.que_classify_multi(q)

            focus = None
            if "focus" in loader.dataset.samples[idx + i]:
                focus = loader.dataset.samples[idx + i]["focus"]

            if len(qtypes) == 0:
                qtypes = [NATOK,]

            for attr in qtypes:
                qtype_hits[attr] += hits[i]
                qtype_counts[attr] += 1

                if focus is None:
                    qtype_hits_without_focus[attr] += hits[i]
                    qtype_counts_without_focus[attr] += 1
                else:
                    qtype_hits_with_focus[attr] += hits[i]
                    qtype_counts_with_focus[attr] += 1

        total_hits += sum(hits)

        total_count += batch_size

        idx += batch_size

    metrics = {
        "acc": total_hits / total_count,
        "count": total_count,
        "qtype_hits": qtype_hits,
        "qtype_hits_with_focus": qtype_hits_with_focus,
        "qtype_hits_without_focus": qtype_hits_without_focus,
        "qtype_counts": qtype_counts,
        "qtype_counts_with_focus": qtype_counts_with_focus,
        "qtype_counts_without_focus": qtype_counts_without_focus,
    }

    return metrics


def run(args):
    # ------------------------------------------------------------------------
    # preamble

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        #torch.set_num_interop_threads(args.num_threads)

    num_workers = 0 if args.num_workers is None else args.num_workers

    # ------------------------------------------------------------------------
    # read arguments from model path

    model_args = ArgumentParser.path_to_args(args.checkpoint)

    # ------------------------------------------------------------------------

    # COCO features loader
    if os.path.exists(os.path.join(args.coco_data_root, "mscoco_bottomup_boxes.npy")):
        coco_data_provider = COCODataProviderSingleFile(args.coco_data_root)
    else:
        coco_data_provider = COCODataProvider(args.coco_data_root)

    # shared tokenizer
    tokenizer = transformers.LxmertTokenizer.from_pretrained(
        "unc-nlp/lxmert-base-uncased"
    )

    set_ = "test"

    # dataset splits with a shared feature provider
    dataset = GuessWhatOracle(
        os.path.join(args.guesswhat_root, "data", f"guesswhat.{set_}.jsonl.gz"),
        os.path.join(args.oracle_targets_root, f"guesswhat_{set_}_oracle.npy"),
        coco_data_provider, tokenizer=tokenizer, max_length=model_args.max_length,
        focus_mode=model_args.focus_mode
    )

    # loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(device.type == "cuda")
    )

    # ------------------------------------------------------------------------

    marker = model_args.marker
    if marker is None:
        marker = "none"

    model = m.LXMERTOracle(marker=marker)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("using {} GPUs".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # ------------------------------------------------------------------------

    if args.checkpoint is not None:  # init backbone with pretrained model
        print("loading checkpoint \"{}\"".format(args.checkpoint))
        checkpoint = torch.load(
            args.checkpoint,
            map_location=lambda storage, loc: storage
        )

        model.load_state_dict(checkpoint["model_state_dict"])

    # ------------------------------------------------------------------------

    table = []

    # question type
    res = eval_ep_qtype(loader, model)
    for qt in sorted(res["qtype_hits"].keys()):
        table.append([
            qt,
            f"{res['qtype_hits'][qt]/(res['qtype_counts'][qt]+2**-23):.4f}",
            f"{res['qtype_hits'][qt]}",
            f"{res['qtype_counts'][qt]}",
            f"{res['qtype_hits_with_focus'][qt]/(res['qtype_counts_with_focus'][qt]+2**-23):.4f}",
            f"{res['qtype_hits_with_focus'][qt]}",
            f"{res['qtype_counts_with_focus'][qt]}",
            f"{res['qtype_hits_without_focus'][qt]/(res['qtype_counts_without_focus'][qt]+2**-23):.4f}",
            f"{res['qtype_hits_without_focus'][qt]}",
            f"{res['qtype_counts_without_focus'][qt]}"
        ])

    print()

    print(f"set={set_}, acc={res['acc']:.3f}, supp={res['count']}")

    print()

    print(tabulate(table, headers=[
        "qtype",
        "acc. (global)", "hits", "supp.",
        "acc. (w/ focus)", "hits", "supp.",
        "acc. (w/o focus)", "hits", "supp."]))

    print()


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Eval GuessWhat?! oracle",
        add_help=True,
        allow_abbrev=False
    )

    parser.add_argument(
        "checkpoint",
        help="checkpoint file",
        type=str
    )

    parser.add_argument(
        "--coco-data-root",
        help="features root path",
        type=str,
        default="./fasterrcnn/mscoco_num-objects_36"
    )

    parser.add_argument(
        "--oracle-targets-root",
        help="oracle targets root path",
        type=str,
        default="./fasterrcnn"
    )

    parser.add_argument(
        "--guesswhat-root",
        help="GuessWhat?! dataset path",
        type=str,
        default="./guesswhat"
    )

    parser.add_argument(
        "--batch-size",
        help="batch size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--num-threads",
        help="torch num threads",
        type=int
    )

    parser.add_argument(
        "--num-workers",
        help="dataloader num workers",
        type=int
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print("{}".format(vars(args)))
    run(args)
