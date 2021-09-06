import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

import transformers


class LXMERTOracle(nn.Module):
    def __init__(self, marker, pretrained=True):
        super(LXMERTOracle, self).__init__()

        if pretrained:
            self._lxmert = transformers.LxmertModel.from_pretrained(
                'unc-nlp/lxmert-base-uncased',
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False
            )
        else:
            self._lxmert = transformers.LxmertModel(
                transformers.LxmertConfig(
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False)
                )

        self.fc = nn.Sequential(
            nn.Linear(768, 3),
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(768, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 3)
        # )

        self.marker = marker

        assert self.marker in ("none", "affine", "offset")

        if self.marker == "affine":

            self.alpha = nn.Parameter(
                torch.FloatTensor(1).fill_(1.0),
                requires_grad=True
            )

            self.beta = nn.Parameter(
                torch.FloatTensor(1).fill_(0.0),
                requires_grad=True
            )

        elif self.marker == "offset":

            self.offset = nn.Parameter(
                torch.FloatTensor(2048),
                requires_grad=True
            )
            torch.nn.init.zeros_(self.offset)

    def forward(self, visdata, langdata):

        # replace last box/feature with target
        tpos = -1

        target_feat = visdata["target_feat"].squeeze()

        if self.marker == "none":
            visdata["visual_feats"][:, tpos] = target_feat
        elif self.marker == "affine":
            visdata["visual_feats"][:, tpos] = self.alpha * target_feat + self.beta
        elif self.marker == "offset":
            visdata["visual_feats"][:, tpos] = target_feat + self.offset

        visdata["visual_pos"][:, tpos] = visdata["target_pos"].squeeze()

        visdata["visual_attention_mask"][:, tpos] = 1.0

        outputs = self._lxmert(
            visual_feats=visdata["visual_feats"],
            visual_pos=visdata["visual_pos"],
            visual_attention_mask=visdata["visual_attention_mask"],
            input_ids=langdata["input_ids"],
            attention_mask=langdata["attention_mask"],
            return_dict=True
        )

        # [CLS] token output
        logits = self.fc(outputs["pooled_output"])

        # # outputs after the cross-modality layers
        # vision_output_mean = outputs.vision_output.sum(1) / (visual_attention_mask.sum(1).unsqueeze(1) + 2**-23)
        # logits = self.fc(vision_output_mean)

        # language_output_mean = outputs.language_output.sum(1) / (langdata["attention_mask"].to(device).sum(1).unsqueeze(1) + 2**-23)
        # logits = self.fc(language_output_mean)

        # # single-modality outputs
        # vision_output_mean = outputs["vision_hidden_states"][0].sum(1) / (visdata["visual_attention_mask"].sum(1).unsqueeze(1) + 2**-23)
        # logits = self.fc(vision_output_mean)

        return logits
