import os

# import json
# import gzip

import numpy as np

# import scipy.io as sio

from PIL import Image

import torch

from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image
from detectron2.structures import Boxes, Instances


class FasterRCNNFeatures(object):
    def __init__(self, root, num_objects: int=36, with_attributes:bool=False, input_min_size: int=800, device:torch.device="cuda") -> None:
        super(FasterRCNNFeatures, self).__init__()

        vocabs = os.path.join(root, "demo", "data", "genome", "1600-400-20")

        self.with_attributes = with_attributes

        self.classes = []
        with open(os.path.join(vocabs, "objects_vocab.txt")) as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())
        MetadataCatalog.get("vg").thing_classes = self.classes

        self.attributes = []
        with open(os.path.join(vocabs, "attributes_vocab.txt")) as f:
            for object in f.readlines():
                self.attributes.append(object.split(',')[0].lower().strip())
        MetadataCatalog.get("vg").attr_classes = self.attributes

        cfg = get_cfg()

        if self.with_attributes:
            cfg.merge_from_file(os.path.join(root, "configs", "VG-Detection", "faster_rcnn_R_101_C4_attr_caffemaxpool.yaml"))
            cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
        else:
            cfg.merge_from_file(os.path.join(root, "configs", "VG-Detection", "faster_rcnn_R_101_C4_caffe.yaml"))
            cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"

        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2

        cfg.INPUT.MIN_SIZE_TEST = int(input_min_size)
        cfg.INPUT.MAX_SIZE_TEST = int(cfg.INPUT.MIN_SIZE_TEST * 1.6667)
        cfg.MODEL.RPN.NMS_THRESH = 0.7

        self.device = device

        cfg["MODEL"]["DEVICE"] = device
        self.predictor = DefaultPredictor(cfg)
        self.predictor.model.eval()

        self.num_objects = num_objects

    @torch.no_grad()
    def __call__(self, raw_image: Image) -> tuple:
        raw_image = np.array(raw_image)[:, :, ::-1]  # RGB->BGR (model pixel mean/std normalization expects BGR format)

        raw_height, raw_width, _ = raw_image.shape
        image = self.predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        height, width, _ = image.shape
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image = image.to(self.device)

        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = self.predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = self.predictor.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = self.predictor.model.proposal_generator(images, features, None)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]

        features = [features[f] for f in self.predictor.model.roi_heads.in_features]
        box_features = self.predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

        if self.with_attributes:
            pred_class_logits, pred_attr_logits, pred_proposal_deltas = self.predictor.model.roi_heads.box_predictor(feature_pooled)
        else:
            pred_class_logits, pred_proposal_deltas = self.predictor.model.roi_heads.box_predictor(feature_pooled)

        outputs = FastRCNNOutputs(
            self.predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]

        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:],
                score_thresh=0.2,
                nms_thresh=nms_thresh,
                topk_per_image=self.num_objects
            )
            if len(ids) == self.num_objects:
                break

        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()

        if self.with_attributes:
            attr_prob = pred_attr_logits[..., :-1].softmax(-1)
            max_attr_prob, max_attr_label = attr_prob.max(-1)
            instances.attr_scores = max_attr_prob[ids].detach()
            instances.attr_classes = max_attr_label[ids].detach()

        return instances, roi_features

    @torch.no_grad()
    def predict(self, raw_image, boxes):
        raw_image = np.array(raw_image)[:, :, ::-1]  # RGB->BGR (model pixel mean/std normalization expects BGR format)

        raw_height, raw_width, _ = raw_image.shape
        image = self.predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)

        # # resize target to 224x224
        # assert boxes.shape[0] == 1
        # boxes_ = boxes.copy()
        # boxes = (boxes[0] + 0.5).astype(np.int32)
        # raw_image = raw_image[boxes[1]:boxes[3]+1, boxes[0]:boxes[2]+1]
        # raw_height, raw_width, _ = raw_image.shape
        # image = np.array(Image.fromarray(raw_image).resize((224, 224), Image.BICUBIC))
        # boxes = np.array([[0, 0, 223, 223]], dtype=np.float32)

        height, width, _ = image.shape
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image = image.to(self.device)

        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = self.predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = self.predictor.model.backbone(images.tensor)

        raw_boxes = torch.from_numpy(np.atleast_2d(boxes))
        assert (
            torch.all(raw_boxes[:, 2] > raw_boxes[:, 0])
            and torch.all(raw_boxes[:, 3] > raw_boxes[:, 1])
            ), "make sure boxes are given in (x1, y1, x2, y2) format"
        raw_boxes = Boxes(raw_boxes.to(self.device))
        scale_x = 1. * width / raw_width
        scale_y = 1. * height / raw_height
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)
        proposal_boxes = [boxes]

        features = [features[f] for f in self.predictor.model.roi_heads.in_features]
        box_features = self.predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])

        if self.with_attributes:
            pred_class_logits, pred_attr_logits, pred_proposal_deltas = self.predictor.model.roi_heads.box_predictor(feature_pooled)
        else:
            pred_class_logits, pred_proposal_deltas = self.predictor.model.roi_heads.box_predictor(feature_pooled)

        pred_class_prob = torch.nn.functional.softmax(pred_class_logits, -1)
        pred_scores, pred_classes = pred_class_prob[..., :-1].max(-1)

        feature_pooled = feature_pooled.detach()

        instances = Instances(
            image_size=(raw_height, raw_width),
            pred_boxes=raw_boxes,
            #pred_boxes=Boxes(torch.from_numpy(np.atleast_2d(boxes_)).to(self.device)),
            scores=pred_scores,
            pred_classes=pred_classes,
        )

        if self.with_attributes:
            attr_prob = pred_attr_logits[..., :-1].softmax(-1)
            max_attr_prob, max_attr_label = attr_prob.max(-1)
            instances.attr_scores = max_attr_prob
            instances.attr_classes = max_attr_label

        return instances, feature_pooled
