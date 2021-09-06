import os

import numpy as np

import torch

import json

import transformers

from utils import load_data, progressbar, xyxy2xywh, xywh2xyxy

from torchvision.ops import box_iou


_ANSWER_TO_LABEL_MAP = {"No": 0, "Yes": 1, "N/A": 2}


class DummyDataProvider(object):
    def __init__(self, num_objects=36):
        self.num_objects = int(num_objects)

    def get_data(self, imkey):
        boxes = np.empty((self.num_objects, 4)).astype(np.float32)
        features = np.empty((self.num_objects, 2048)).astype(np.float32)
        return boxes, features


class COCODataProvider(object):
    def __init__(self, root, num_objects=36):
        super(COCODataProvider, self).__init__()
        self.root = os.path.abspath(root)
        self.num_objects = int(num_objects)

    def get_data(self, imkey):
        coco_set = imkey.split("_")[1]
        imfile = os.path.join(
            self.root, coco_set, os.path.splitext(imkey)[0] + ".npy"
        )
        data = np.load(imfile, allow_pickle=True).item()
        boxes, features = data["boxes"], data["features"]

        if len(boxes) == 0:
            boxes = np.zeros((self.num_objects, 4), dtype=np.float32)
            features = np.zeros((self.num_objects, 2048), dtype=np.float32)
        elif boxes.shape[0] < self.num_objects:
            n = self.num_objects - boxes.shape[0]
            boxes = np.vstack([boxes, np.zeros((n, 4), dtype=np.float32)])
            features = np.vstack([features, np.zeros((n, 2048), dtype=np.float32)])
        else:
            boxes = boxes[:self.num_objects]
            features = features[:self.num_objects]

        return boxes, features


class COCODataProviderSingleFile(object):
    def __init__(self, root, num_objects=36):
        self.boxes = np.load(
            os.path.join(root, "mscoco_bottomup_boxes.npy")
        )  # in (x1, y1, x2, y2) format

        self.features = np.load(
            os.path.join(root, "mscoco_bottomup_features.npy"),
            mmap_mode="c"
        )

        self.info = json.load(open(
            os.path.join(root, "mscoco_bottomup_info.json")
        ))

        assert num_objects == 36

    def get_data(self, imkey):
        if imkey.endswith(".jpg"):
            imkey = os.path.splitext(imkey)[0]
        pos = self.info["image_id2image_pos"][imkey]
        features = self.features[pos]
        boxes = xyxy2xywh(self.boxes[pos])  # (x,y,w,h) by default
        return boxes, features


class GuessWhatOracle(torch.utils.data.Dataset):
    def __init__(self,
                 guesswhat_games_file,
                 oracle_targets_file,
                 coco_data_provider,
                 success_only=True,
                 tokenizer=None,
                 max_length=20,
                 focus_mode="none"):
        super(GuessWhatOracle, self).__init__()

        self.samples = []

        for g in progressbar(load_data(guesswhat_games_file), desc="loading games"):
            if success_only and g["status"] != "success":
                continue

            # target = [
            #     obj["bbox"]
            #     for obj in g["objects"]
            #     if obj["id"] == g["object_id"]
            # ]

            gid = g["id"]
            file_name = g["image"]["file_name"]
            height = g["image"]["height"]
            width = g["image"]["width"]

            self.samples += [
                {
                    "gid": gid,
                    "image_file": file_name,
                    "image_height": height,
                    "image_width": width,
                    "question": qa["question"],
                    "qid": qa["id"],
                    "answer": _ANSWER_TO_LABEL_MAP[qa["answer"]],
                    "pos": pos,
                    "length": len(g["qas"]),
                    "focus": None if "focus" not in qa else qa["focus"]
                }
                for pos, qa in enumerate(g["qas"])
            ]

        self.coco_data_provider = coco_data_provider

        self.targets = np.load(oracle_targets_file, allow_pickle=True).item()

        # keys are originally strings because of the .npy format
        self.targets = {int(k): v for k, v in self.targets.items()}

        # # debuging
        # p_samples = 0.05
        # idxs = np.random.permutation(len(self.samples))[:int(p_samples*len(self.samples))]
        # self.samples = [self.samples[i] for i in idxs]

        self.tokenizer = transformers.LxmertTokenizer.from_pretrained(
            'unc-nlp/lxmert-base-uncased'
        ) if tokenizer is None else tokenizer

        self.max_length = int(max_length)

        self.focus_mode = focus_mode
        assert self.focus_mode in ("none", "relative", "restriction", "mixed", "zeros", "random")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # read detections
        boxes, features = self.coco_data_provider.get_data(sample["image_file"])
        boxes = torch.from_numpy(boxes)

        boxes = xywh2xyxy(boxes)

        features = torch.from_numpy(features)

        # get target data
        target = self.targets[sample["gid"]]

        target_box = target["boxes"].copy()
        target_box = torch.from_numpy(target_box)

        target_box = xywh2xyxy(target_box)

        target_feature = target["features"].copy()
        target_feature = torch.from_numpy(target_feature)

        # mask out empty detections
        valid_boxes = (boxes.abs().sum(-1) > 1e-4).float()

        width, height = sample["image_width"], sample["image_height"]

        # set focus
        focus = sample["focus"]
        if focus is None:  # if not set, use whole image
            focus = (0, 0, width, height)

        if self.focus_mode == "none":
            # normalize to the image size

            boxes[:, (0, 2)] /= width
            boxes[:, (1, 3)] /= height

            target_box[0, (0, 2)] /= width
            target_box[0, (1, 3)] /= height

        elif self.focus_mode == "relative":
            # normalize coordinates to the focus region

            boxes[:, (0, 2)] = (boxes[:, (0, 2)] - focus[0]) / focus[2]
            boxes[:, (1, 3)] = (boxes[:, (1, 3)] - focus[1]) / focus[3]

            target_box[0, (0, 2)] = (target_box[0, (0, 2)] - focus[0]) / focus[2]
            target_box[0, (1, 3)] = (target_box[0, (1, 3)] - focus[1]) / focus[3]

        elif self.focus_mode == "restriction":
            # mask out boxes that not intersect with the focus.
            # normalize to the image size

            focus_ = xywh2xyxy(
                torch.tensor(focus, dtype=torch.float).unsqueeze(0)
            )
            nonzero_iou = (box_iou(focus_, boxes).squeeze() > 1e-4).float()
            valid_boxes *= nonzero_iou

            boxes[:, (0, 2)] /= width
            boxes[:, (1, 3)] /= height

            target_box[0, (0, 2)] /= width
            target_box[0, (1, 3)] /= height

        elif self.focus_mode == "mixed":
            # mask out boxes that not intersect with the focus.
            # normalize coordinates to the focus region

            focus_ = xywh2xyxy(
                torch.tensor(focus, dtype=torch.float).unsqueeze(0)
            )
            nonzero_iou = (box_iou(focus_, boxes).squeeze() > 1e-4).float()
            valid_boxes *= nonzero_iou

            boxes[:, (0, 2)] = (boxes[:, (0, 2)] - focus[0]) / focus[2]
            boxes[:, (1, 3)] = (boxes[:, (1, 3)] - focus[1]) / focus[3]

            target_box[0, (0, 2)] = (target_box[0, (0, 2)] - focus[0]) / focus[2]
            target_box[0, (1, 3)] = (target_box[0, (1, 3)] - focus[1]) / focus[3]

        elif self.focus_mode == "zeros":
            # zeroed coordinates

            target_box *= 0.0
            boxes *= 0.0

        elif self.focus_mode == "random":
            # sample normalized (x, y, w, h) coordinates

            target_box = torch.rand_like(target_box)
            boxes = torch.rand_like(boxes)

        else:
            raise ValueError("not a valid mode")

        visdata = {
            "visual_feats": features,
            "visual_pos": boxes,
            "visual_attention_mask": valid_boxes,
            "target_feat": target_feature,
            "target_pos": target_box
        }

        langdata = self.tokenizer(
            sample["question"],
            return_tensors="pt",
            padding="max_length",
            return_token_type_ids=False,
            return_attention_mask=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length
        )
        langdata["input_ids"] = langdata["input_ids"].squeeze()
        langdata["attention_mask"] = langdata["attention_mask"].squeeze()

        answer = torch.LongTensor([sample["answer"]])

        return visdata, langdata, answer
