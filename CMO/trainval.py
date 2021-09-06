import os

import sys

import signal

import argparse

import numpy as np

import torch

from torch.utils.tensorboard import SummaryWriter

from utils import load_data, progressbar, hms, cprint

from datasets import COCODataProvider, COCODataProviderSingleFile, GuessWhatOracle

from parser import ArgumentParser

import models as m

import transformers



def train_ep(loader, model, criterion, optimizer, scheduler=None,
             max_grad_norm=0.0, description=""):
    device = next(model.parameters()).device

    model.train()

    total_loss = 0.0
    hits = 0

    n_samples = 0

    for visdata, langdata, targets in progressbar(loader, desc=description):
        for k, v in visdata.items():
            visdata[k] = v.to(device)

        for k, v in langdata.items():
            langdata[k] = v.to(device)

        targets = targets.squeeze(-1).to(device)

        logits = model(visdata, langdata)

        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * targets.size(0)

        preds = logits.detach().argmax(1)

        hits += (preds == targets).float().sum().cpu().item()

        n_samples += targets.size(0)

    metrics = {
        "acc": hits / n_samples,
        "loss": total_loss / n_samples,
    }

    return metrics


@torch.no_grad()
def eval_ep(loader, model, criterion=None, description=""):
    device = next(model.parameters()).device

    model.eval()

    total_loss = 0.0
    hits = 0

    n_samples = 0

    for visdata, langdata, targets in progressbar(loader, desc=description):

        for k, v in visdata.items():
            visdata[k] = v.to(device)

        for k, v in langdata.items():
            langdata[k] = v.to(device)

        targets = targets.squeeze(-1).to(device)

        logits = model(visdata, langdata)

        if criterion is not None:
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)

        preds = logits.detach().argmax(1)

        hits += (preds == targets).float().sum().cpu().item()

        n_samples += targets.size(0)

    metrics = {
        "acc": hits / n_samples,
        "loss": total_loss / n_samples,
    }

    return metrics


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

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        #torch.set_num_interop_threads(args.num_threads)

    num_workers = 0 if args.num_workers is None else args.num_workers

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

    # dataset splits with a shared feature provider
    datasets = {
        set_: GuessWhatOracle(
            os.path.join(args.guesswhat_root, "data", f"guesswhat.{set_}.jsonl.gz"),
            os.path.join(args.oracle_targets_root, f"guesswhat_{set_}_oracle.npy"),
            coco_data_provider, tokenizer=tokenizer, max_length=args.max_length,
            focus_mode=args.focus_mode
        ) for set_ in ("train", "valid", "test")
    }

    # loaders
    loaders = {
        set_: torch.utils.data.DataLoader(
            datasets[set_],
            batch_size=args.batch_size,
            shuffle=bool(set_ == "train"),
            num_workers=num_workers,
            pin_memory=bool(device.type == "cuda")
        ) for set_ in datasets.keys()
    }

    # ------------------------------------------------------------------------

    model = m.LXMERTOracle(marker=args.marker, pretrained=True)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("using {} GPUs".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # optimizer = torch.optim.Adam(
    #     [p for p in model.parameters() if p.requires_grad],
    #     lr=args.learning_rate, weight_decay=args.weight_decay
    # )
    # scheduler = None

    optimizer = transformers.optimization.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
        correct_bias=False
    )
    scheduler = None

    # # https://huggingface.co/transformers/migration.html (at the end)
    # num_training_steps = len(loaders["train"]) * args.epochs
    # num_warmup_steps = int(0.1 * num_training_steps)
    # scheduler = transformers.get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_training_steps
    # )

    # lr_factor = 0.1
    # features_param = [p for name, p in model.named_parameters() if p.requires_grad and "fc." not in name]
    # classifier_param = [p for name, p in model.named_parameters() if p.requires_grad and "fc." in name]
    # optimizer = transformers.optimization.AdamW([
    #     {"params": features_param, "lr": lr_factor * args.learning_rate},
    #     {"params": classifier_param, "lr": args.learning_rate}
    # ], lr=args.learning_rate, weight_decay=args.weight_decay, correct_bias=False)
    # scheduler = None

    # ------------------------------------------------------------------------

    epoch0 = 1

    CHECKPOINTING_STEP = max(5, args.epochs // 10)

    VALIDATION_STEP = 1

    val_metric = "acc"

    if args.checkpoint is not None:  # init backbone with pretrained model
        print("loading checkpoint \"{}\"".format(args.checkpoint))
        checkpoint = torch.load(
            args.checkpoint,
            map_location=lambda storage, loc: storage
        )

        epoch0 = checkpoint["epoch"] + 1

        model.load_state_dict(checkpoint["model_state_dict"])

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # ------------------------------------------------------------------------

    output_dir = ArgumentParser.args_to_path(args)
    os.makedirs(output_dir, exist_ok=True)

    # log arguments for future reference
    with open(output_dir + ".log", "w") as fh:
        fh.write(f"{vars(args)}")

    writer = SummaryWriter(output_dir)

    def log_metrics(set_, epoch, metrics, color=None):
        cprint((
            f"{hms()} [{epoch}/{args.epochs}] {set_}: "
            f"loss={metrics['loss']:.3e}, "
            f"acc={metrics['acc']:.3f}"
        ), color=color)
        writer.add_scalar(f"loss/{set_}", metrics["loss"], epoch)
        writer.add_scalar(f"acc/{set_}", metrics["acc"], epoch)
        writer.flush()

    def test_best_model():
        model_file = os.path.join(output_dir, "best.pt")
        if not os.path.exists(model_file):
            return
        checkpoint = torch.load(
            model_file, map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics = eval_ep(
            loaders["test"], model, description=f"{hms()} test"
        )
        log_metrics("test", checkpoint["epoch"], metrics, color="blue")

    def test_on_exit(sig, frame):
        test_best_model()
        sys.exit(0)

    signal.signal(signal.SIGINT, test_on_exit)  # register Ctrl+C handler

    best_value = 0.0

    for epoch in range(epoch0, args.epochs+1):

        metrics = train_ep(
            loaders["train"],
            model, criterion, optimizer, scheduler,
            max_grad_norm=args.max_grad_norm,
            description=f"{hms()} [{epoch}/{args.epochs}] train"
        )
        log_metrics("train", epoch, metrics)

        if epoch % CHECKPOINTING_STEP == 0:
            # fname = os.path.join(output_dir, f"{epoch:04d}.pt")
            fname = os.path.join(output_dir, f"last.pt")
            save_checkpoint(fname, epoch, model, optimizer, scheduler)

        if (epoch == epoch0) or (epoch % VALIDATION_STEP == 0):

            metrics = eval_ep(
                loaders["valid"], model, criterion,
                description=f"{hms()} [{epoch}/{args.epochs}] valid"
            )

            if metrics[val_metric] > best_value:
                best_value = metrics[val_metric]
                fname = os.path.join(output_dir, "best.pt")
                save_checkpoint(fname, epoch, model, optimizer, scheduler)

                log_metrics("valid", epoch, metrics, color="green")

            else:
                log_metrics("valid", epoch, metrics)

        # if scheduler is not None:
        #     scheduler.step()

    test_best_model()


def save_checkpoint(fname, epoch, model, optimizer, scheduler=None):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, fname)


if __name__ == "__main__":
    args = ArgumentParser(
        description="LXMERT Oracle baseline",
        with_data_params=True,
        with_model_params=True,
        with_learning_params=True,
        with_dialog_history_params=True
    ).parse_args()
    cprint(f"{vars(args)}", color="red")
    run(args)
