import sys

import numpy as np
import datetime
import json
import argparse
import os
import multiprocessing
from collections import OrderedDict
from tensorboardX import SummaryWriter
from time import time

from utils.config import load_config
from utils.vocab import create_vocab
from utils.datasets.Oracle.OracleDataset import OracleDataset
from models.Oracle import Oracle

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/Oracle/config.json", help='Config file')
    parser.add_argument("-img_feat", type=str, default="rss", help='Select "vgg" or "res" as image features')

    args = parser.parse_args()

    config = load_config(args.config)

    # Hyperparamters
    data_paths          = config['data_paths']
    optimizer_config    = config['optimizer']
    embedding_config    = config['embeddings']
    lstm_config         = config['lstm']
    mlp_config          = config['mlp']
    dataset_config      = config['dataset']
    inputs_config       = config['inputs']

    # Experiment Settings
    exp_config = config['exp_config']
    exp_config['img_feat'] = args.img_feat.lower()

#    dataset_train = OracleDataset(
#        data_dir            = args.data_dir,
#        data_file           = data_paths['train_file'],
#        split               = 'train',
#        visual_feat_file    = data_paths[args.img_feat]['image_features'],
#        visual_feat_mapping_file = data_paths[exp_config['img_feat']]['img2id'],
#        visual_feat_crop_file = data_paths[args.img_feat]['crop_features'],
#        visual_feat_crop_mapping_file = data_paths[exp_config['img_feat']]['crop2id'],
#        max_src_length      = dataset_config['max_src_length'],
#        hdf5_visual_feat    = 'train_img_features',
#        hdf5_crop_feat      = 'objects_features',
#        history             = dataset_config['history'],
#        new_oracle_data     = True,
#        successful_only     = dataset_config['successful_only'],
#        load_crops          = inputs_config['crop'],
#        record_history      = dataset_config['record_history'],
#        max_hlen            = dataset_config['max_hlen'],
#        keep_yes_only       = dataset_config['keep_yes_only']
#    )
#
    dataset_validation = OracleDataset(
        data_dir            = args.data_dir,
        data_file           = data_paths['val_file'],
        split               = 'val',
        visual_feat_file    = data_paths[args.img_feat]['image_features'],
        visual_feat_mapping_file = data_paths[exp_config['img_feat']]['img2id'],
        visual_feat_crop_file = data_paths[args.img_feat]['crop_features'],
        visual_feat_crop_mapping_file = data_paths[exp_config['img_feat']]['crop2id'],
        max_src_length      = dataset_config['max_src_length'],
        hdf5_visual_feat    = 'val_img_features',
        hdf5_crop_feat      = 'objects_features',
        history             = dataset_config['history'],
        new_oracle_data     = True,
        successful_only     = False,
        load_crops          = inputs_config['crop'],
        record_history      = dataset_config['record_history'],
        max_hlen            = dataset_config['max_hlen'],
        keep_yes_only       = dataset_config['keep_yes_only']
    )

    dataset_test = OracleDataset(
        data_dir            = args.data_dir,
        data_file           = data_paths['test_file'],
        split               = 'test',
        visual_feat_file    = data_paths[args.img_feat]['image_features'],
        visual_feat_mapping_file = data_paths[exp_config['img_feat']]['img2id'],
        visual_feat_crop_file = data_paths[args.img_feat]['crop_features'],
        visual_feat_crop_mapping_file = data_paths[exp_config['img_feat']]['crop2id'],
        max_src_length      = dataset_config['max_src_length'],
        hdf5_visual_feat    = 'test_img_features',
        hdf5_crop_feat      = 'objects_features',
        history             = dataset_config['history'],
        new_oracle_data     = True,
        successful_only     = False,
        load_crops          = inputs_config['crop'],
        record_history      = dataset_config['record_history'],
        max_hlen            = dataset_config['max_hlen'],
        keep_yes_only       = dataset_config['keep_yes_only']
    )

