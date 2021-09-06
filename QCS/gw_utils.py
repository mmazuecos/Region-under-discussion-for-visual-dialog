import datetime
import gzip
import json
import wget
import jsonlines
import h5py
import os

import numpy as np
import pandas as pd
from time import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.config import load_config
from utils.model_loading import load_model
from utils.image_utils import get_spatial_feat
#from utils.datasets.Oracle.OracleDataset import OracleDataset
from models.Oracle import Oracle
from urllib.error import HTTPError

IMG_DIR = './viz/'

class Model():
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward(self, sample):
        questions, _, crop_features, visual_features, spatials, obj_categories, lengths = \
            sample['question'], sample['answer'], sample['crop_features'], sample['img_features'], sample['spatial'], sample['obj_cat'], sample['length']

        # Forward pass
        pred_answer = self.model(Variable(questions),
            Variable(obj_categories),
            Variable(spatials),
            Variable(crop_features),
            Variable(visual_features),
            Variable(lengths)
        )
        return pred_answer


class Vocab_Manager():
    """
    The whole transformation from tokens to words and vice versa in a single
    class.
    """
    def __init__(self, oracle_args):
        with open(os.path.join('data', oracle_args['data_paths']['vocab_file'])) as file:
            vocab = json.load(file)
        self.word2i = vocab['word2i']
        self.i2word = vocab['i2word']
        self.vocab_size = len(self.word2i)

    def seq2words(self, sequence):
        sentence = ''
        for token in sequence:
            sentence += ' '+ self.i2word[str(token)] 
        return sentence

    def words2seq(self, sentence):
        sequence = []
        for word in sentence.lower().strip().split(' '):
            if word not in self.word2i.keys():
                token = 4
            else:
                token = self.word2i[word]
            sequence.append(token)

        sequence = sequence + [0 for _ in range(15 - len(sequence))]
        return sequence


def get_config(img_feat='rss'):
    config = load_config('config/Oracle/config.json')
    exp_config = config['exp_config']
    exp_config['img_feat'] = img_feat
    exp_config['use_cuda'] = torch.cuda.is_available()
    exp_config['ts'] = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H_%M'))
    return config, exp_config


def load_oracle(oracle_bin_path, oracle_args, vocab_size):

    oracle = Oracle(
        no_words            = vocab_size,
        no_words_feat       = oracle_args['embeddings']['no_words_feat'],
        no_categories       = oracle_args['embeddings']['no_categories'],
        no_category_feat    = oracle_args['embeddings']['no_category_feat'],
        no_hidden_encoder   = oracle_args['lstm']['no_hidden_encoder'],
        mlp_layer_sizes     = oracle_args['mlp']['layer_sizes'],
        no_visual_feat      = oracle_args['inputs']['no_visual_feat'],
        no_crop_feat        = oracle_args['inputs']['no_crop_feat'],
        dropout             = oracle_args['lstm']['dropout'],
        inputs_config       = oracle_args['inputs'],
        scale_visual_to     = oracle_args['inputs']['scale_visual_to']
        )

    oracle = load_model(oracle, oracle_bin_path,
            use_dataparallel=False)
    
    return oracle


def load_dataset(oracle_args, exp_config, split='train'):
    dataset_train = OracleDataset(
        data_dir            = 'data',
        data_file           = oracle_args['data_paths']['train_file'],
        split               = split,
        visual_feat_file    = oracle_args['data_paths']['rss']['image_features'],
        visual_feat_mapping_file = oracle_args['data_paths'][exp_config['img_feat']]['img2id'],
        visual_feat_mapping_type = 'train2id',
        visual_feat_crop_file = oracle_args['data_paths']['rss']['crop_features'],
        visual_feat_crop_mapping_file = oracle_args['data_paths'][exp_config['img_feat']]['crop2id'],
        max_src_length      = oracle_args['dataset']['max_src_length'],
        hdf5_visual_feat    = 'train_img_features',
        hdf5_crop_feat      = 'crop_features',
        history             = oracle_args['dataset']['history'],
        new_oracle_data     = False,
        successful_only     = oracle_args['dataset']['successful_only']
    )
    return dataset_train 


def get_raw_dataset(data_split):
    return jsonlines.Reader(gzip.open(data_split))


def mk_dataloader(dataset, exp_config):
    dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=0,
            num_workers=1,
            pin_memory=exp_config['use_cuda']
            )
    return enumerate(dataloader)


def get_obj_data(game):
    game_objects = game['objects']
    im_width = game['image']['width']
    im_height = game['image']['height']
    obj_data = []
    for obj in game_objects:
        data_tuple = (obj['id'],
                      obj['category_id'],
                      get_spatial_feat(obj['bbox'], im_width, im_height)
                      )
        obj_data.append(data_tuple)
    return obj_data


def tuple2tensor(tup):
    return (torch.tensor([tup[1]]), torch.tensor([tup[2]]))


def load_data():
    with open('data/ResNet_avg_image_features2id.json', 'rb') as fl:
        vf_map = json.load(fl)
        vf_map = vf_map['test2id']
    with gzip.open('data/guesswhat.test.jsonl.gz', 'r') as fl:
        games = [entry for entry in jsonlines.Reader(fl)]
    visual_features = np.asarray(h5py.File('data/ResNet_avg_image_features.h5',
        'r')['test_img_features'])
    return games, vf_map, visual_features


def ask_with_string(string, game, vf_map, visual_feat ,vocab, oracle):
    question = ''
    hist = []

    answers = {0:'No', 1: 'Yes', 2: 'N/A'}

    img_feat_idx = vf_map[game['image']['file_name']]
    img_feat = visual_feat[img_feat_idx]

    target_obj_id = game['object_id']
    
    for o in game['objects']:
        if o['id'] == target_obj_id:
            target = o
        else:
            continue

    im_width = game['image']['width']
    im_height = game['image']['height']

    sample = {}
    sample['answer']         = torch.tensor([2])
    sample['crop_features']  = torch.tensor([0])
    sample['img_features'] = torch.tensor([img_feat])
    sample['spatial'] = torch.tensor([get_spatial_feat(target['bbox'],
                                                      im_width,
                                                      im_height)])
    sample['obj_cat'] = torch.tensor([target['category_id']])

    question = string
    qseq = vocab.words2seq(question)
    preproq = vocab.seq2words(qseq)
    sample['length'] = torch.tensor([len(question.strip().split(' '))])
    sample['question'] = torch.tensor([qseq])
    
    #print(sample)

    pred = oracle.forward(sample).argmax(dim=1).numpy()
    
    return pred[0] 


def load_vocab_oracle(oracle_bin_path):
    config, exp_config = get_config()
    vocab = Vocab_Manager(config)
    oracle = load_oracle(oracle_bin_path, config, vocab.vocab_size)
    inf_model = Model(oracle)
    return vocab, inf_model


def guess(game, img):
    objects = game['objects']

    target_id = game['object_id']

    print('Objects:')
    for i, obj in enumerate(objects):
        # Every game has an object whose id is target_id
        # so it's safe the assign target this way
        if obj['id'] == target_id:
            target = obj
        print(i, obj['category'])

    #plt.show()
    #plt.clf()

    valid = False
    while not valid:
        answer = input('Which object is it? ')
        if int(answer) < 0 and int(answer) > len(game['objects']):
            print('Please enter a valid option')
        else:
            valid = True

    if objects[int(answer)]['id'] == target_id:
        print("CONGRATULATIONS! You guessed the target object!")
    else:
        print("OOPS, it was not the right object")

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    tbbox = target['bbox']
    trect = patches.Rectangle((tbbox[0], tbbox[1]),
                              tbbox[2],
                              tbbox[3],
                              linewidth=4,
                              edgecolor='g', facecolor='none')
    
    ax.add_patch(trect)

def play(game, vf_map, visual_feat ,vocab, oracle):
    question = ''
    hist = []

    answers = {0:'No', 1: 'Yes', 2: 'N/A'}

    img_feat_idx = vf_map[game['image']['file_name']]
    img_feat = visual_feat[img_feat_idx]

    target_obj_id = game['object_id']
    
    for o in game['objects']:
        if o['id'] == target_obj_id:
            target = o
        else:
            continue

    im_width = game['image']['width']
    im_height = game['image']['height']

    sample = {}
    sample['answer']         = torch.tensor([2])
    sample['crop_features']  = torch.tensor([0])
    sample['img_features'] = torch.tensor([img_feat])
    sample['spatial'] = torch.tensor([get_spatial_feat(target['bbox'],
                                                      im_width,
                                                      im_height)])
    sample['obj_cat'] = torch.tensor([target['category_id']])


    while True:
        question = input('Question: ')
        if question == 'END':
            break
        qseq = vocab.words2seq(question)
        sample['length'] = torch.tensor([len(question.strip().split(' '))])
        sample['question'] = torch.tensor([qseq])
        
        #print(sample)

        pred = oracle.forward(sample).argmax(dim=1).numpy()
        
        print('Answer: {}'.format(answers[pred[0]]))


def get_game(games, game_index=3):
    #game_index = np.random.randint(0, high=len(games))
    #print(game_index)
    game = games[game_index]
    img = Image.open(os.path.join(IMG_DIR, game['image']['file_name']))
    osize = (game['image']['width'], game['image']['height'])
    img = img.resize((osize), resample=Image.NEAREST)

    print("Assigned game {}\n".format(game['id']))

    objects = game['objects']
    fig, ax = plt.subplots(1)

    ax.imshow(img)

    im_height = game['image']['height']
    target_id = game['object_id']

    print('Objects:')
    for i, obj in enumerate(objects):
        # Every game has an object whose id is target_id
        # so it's safe the assign target this way
        if obj['id'] == target_id:
            target = obj
        print(i, obj['category'])
        # bbox = [left, up, width, heigh]
        bbox = obj['bbox']
        if obj['id'] == target_id:
            target = obj
        x_width = bbox[2]
        y_height = bbox[3]

        x_left = bbox[0]
        x_right = x_left + x_width

        y_upper = bbox[1]
        y_lower = y_upper - y_height
        rect = patches.Rectangle((x_left, y_upper),
                                 x_width,
                                 y_height,
                                 linewidth=1,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_right, y_upper-1, str(i), color='r', fontsize=34)

    plt.show()


    return game, img

def plot_game(game):
    #game_index = np.random.randint(0, high=len(games))
    #print(game_index)
    if not os.path.exists(os.path.join(IMG_DIR, game['image']['file_name'])):
        img_fname = wget.download(game['image']['flickr_url'])
        os.rename(img_fname, os.path.join(IMG_DIR, game['image']['file_name']))
    img = Image.open(os.path.join(IMG_DIR, game['image']['file_name']))
    osize = (game['image']['width'], game['image']['height'])
    img = img.resize((osize), resample=Image.NEAREST)

    print("Assigned game {}\n".format(game['id']))

    objects = game['objects']
    fig, ax = plt.subplots(1)

    ax.imshow(img)

    im_height = game['image']['height']
    target_id = game['object_id']

    print('Objects:')
    for i, obj in enumerate(objects):
        # Every game has an object whose id is target_id
        # so it's safe the assign target this way
        if obj['id'] == target_id:
            target = obj
        print(i+1, obj['category'])
        # bbox = [left, up, width, heigh]
        bbox = obj['bbox']
        if obj['id'] == target_id:
            target = obj
        x_width = bbox[2]
        y_height = bbox[3]

        x_left = bbox[0]
        x_right = x_left + x_width

        y_upper = bbox[1]
        y_lower = y_upper - y_height
        if obj['id'] == target_id:
            rect = patches.Rectangle((x_left, y_upper),
                                     x_width,
                                     y_height,
                                     linewidth=1.8,
                                     edgecolor='g', facecolor='none')
        else:
            rect = patches.Rectangle((x_left, y_upper),
                                     x_width,
                                     y_height,
                                     linewidth=1.8,
                                     edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if obj['id'] == target_id:
            ax.text(x_right, y_upper-1, str(i+1), color='g', fontsize=34)
        else:
            ax.text(x_right, y_upper-1, str(i+1), color='r', fontsize=34)

    plt.show()

if __name__ == '__main__':
    #config, exp_config, vocab, oracle, dataloader, rawdata = load_all()
    rawdata = get_raw_dataset('data/guesswhat.train.jsonl.gz')

    #if not os.path.isfile('igames-annotations.json'):
    interesting_games = []
    while len(interesting_games) < 50:
        u = np.random.rand()
        game = rawdata.read()
        if u > 0.6:
            try:
                filename = wget.download(game['image']['flickr_url'])
                os.rename(filename, 'image_'+str(game['id'])+'.jpg')
                interesting_games.append(game)
            except HTTPError:
                print('Game {} failed: picture not found'.format(game['id']))
                continue
        else:
            continue
    
    with open('data/igames-annotations.json', 'w') as fl:
        jsonwriter = jsonlines.Writer(fl)
        for game in interesting_games:
            jsonwriter.write(game)
        jsonwriter.close()
    #else:
    #    with open('saved_igames.json', 'r') as fl:
    #        interesting_games = [entry for entry in jsonlines.Reader(fl)]

