import argparse

import json

from utils import load_data, save_data, progressbar

import os
import io
import json
import gzip
import argparse
import collections
from time import time
from nltk.tokenize import TweetTokenizer
import statistics
import copy

# adapted from:
#   https://github.com/shekharRavi/Beyond-Task-Success-NAACL2019/blob/master/analysis/qclassify.py
#
# data/word_annotation can be downloaded from:
#   https://github.com/shekharRavi/Beyond-Task-Success-NAACL2019/blob/master/data/word_annotation

class QTypeClassifier():
    def __init__(self, word_annotations="data/word_annotation"):
        super(QTypeClassifier, self).__init__()

        self.tknzr = TweetTokenizer(preserve_case=False)

        self.qtype = {
            "color": [],
            "shape": [],
            "size": [],
            "texture": [],
            "action": [],
            "activity": [],
            "spatial": [],
            "number": [],
            "object": [],
            "super-category": [],
        }

        with open(word_annotations) as f:
            for line in f.readlines():
                word1, word2 = line.split('\t')
                word1 = word1.strip().lower()
                word2 = word2.strip().lower()
                self.qtype[word2].append(word1)

        # add spatial OR number category
        self.qtype["spatial|number"] = self.qtype["spatial"] + self.qtype["number"]

        # remove activity
        del self.qtype["activity"]

        for k, v in self.qtype.items():
            self.qtype[k] = set(v)

        self.attributes = set(self.qtype.keys())
        self.attributes.remove("object")
        self.attributes.remove("super-category")

    def que_classify_single(self, que):
	# To Classify based on Attribute, Object and Super-category
        tokens = set(self.tknzr.tokenize(que.lower()))

        cat = "N/A"

        for key in self.attributes:
            if cat == "N/A" and len(tokens.intersection(self.qtype[key])) > 0:
                cat = "attribute"

        if cat == "N/A" and len(tokens.intersection(self.qtype["object"])) > 0:
            cat = "object"

        if cat == "N/A" and len(tokens.intersection(self.qtype["super-category"])) > 0:
            cat = "super-category"

        return  cat

    def que_classify_multi(self, que):
	# Question Classification
        tokens = set(self.tknzr.tokenize(que.lower()))

        cats = []
        for key in self.attributes:
            if len(tokens.intersection(self.qtype[key])) > 0:
                cats.append(key)

        if cats == [] and len(tokens.intersection(self.qtype["object"])) > 0:
            cats = ["object",]

        if cats == [] and len(tokens.intersection(self.qtype["super-category"])) > 0:
            cats = ["super-category",]

        return cats


# def run(args):

#     games = load_data(args.guesswhat_games)

#     clf = QTypeClassifier(args.word_annotations)

#     qtype_map = {}

#     for i, g in progressbar(enumerate(games), total=len(games)):
#         for qa in g["qas"]:
#             if qa["id"] in qtype_map:
#                 continue
#             qtype = clf.que_classify_multi(qa["question"])
#             qtype_map[qa["id"]] = qtype

#     qtype_map["attributes"] = sorted(list(clf.attributes))
#     qtype_map["qtypes"] = sorted(list(clf.qtype.keys()))

#     with open(args.output_file, "w") as f:
#         json.dump(qtype_map, f)

#     print(f"{args.output_file} saved")


# def parse_arguments():
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#         description="Compute question type map from GuessWhat?! data",
#         add_help=True,
#         allow_abbrev=False
#     )

#     parser.add_argument(
#         "guesswhat_games",
#         help="GuessWhat?! games file",
#         type=str
#     )

#     parser.add_argument(
#         "--word-annotations",
#         help="word annotations file",
#         type=str,
#         default="./data/word_annotation"
#     )

#     parser.add_argument(
#         "--output-file",
#         help="output file",
#         type=str,
#         default="./output.json"
#     )

#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_arguments()
#     print("{}".format(vars(args)))
#     run(args)
