import gzip
import json
import wget
import jsonlines
import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from PIL import Image

IMG_DIR = './viz/'

class Fetcher():
    """
    Game fetcher.
    
    Parameters:
    -----------
    
    data_reader: jsonlines.Reader
        Reader with the dataset.
    """
    def __init__(self, data_reader):
        self.loaded_data = {}
        self.loader = data_reader
        
    def fetch(self, key):
        """
        Fetch a game in the cache or find it in the data_reader
        
        Parameters:
        -----------
        
        key: int
            game_id to search for in the data.
        """
        try:
            return self.loaded_data[key]
        except KeyError:
            nkey = None
            while nkey != key:
                new_entry = self.loader.read()
                nkey = new_entry['id']
                self.loaded_data[new_entry['id']] = new_entry
            return new_entry

        
def pretty_show(pgame, game):
    """
    Print a generated dialogue the pretty way
    
    Parameters:
    -----------
    
    pgame: dict
        A generated dialogue with effectiveness and RS computed
        
    game: dict
        An entry of the GuessWhat?! dataset
    """
    print('\n')
    id2id = dict([(obj['id'],i) for i, obj in enumerate(game['objects'])])
    if 'model_ans' in pgame['qas'][0].keys():
        for entry in pgame['qas']: 
            idrs = [id2id[ent] for ent in entry['rs']]
            print('{} --- {} (model: {}) RS: {} Effective: {} RMT {}'.format(entry['question'],
                                                                             entry['ans'],
                                                                             entry['model_ans'],
                                                                             idrs,
                                                                             entry['effective'],
                                                                             entry['removed_target']))
    else:
        for entry in pgame['qas']: 
            idrs = [id2id[ent] for ent in entry['rs']]
            print('{} --- {} RS: {} Effective: {}'.format(entry['question'], entry['ans'], idrs, entry['effective']))
    print('-----------\n')


def plot_game(game):
    """
    Plot the image, with its respective objects in bounding boxes and the target
    objects marked.

    Parameters:
    -----------

    game: dict 
        An entry of the GuessWhat?! dataset
    """
    #game_index = np.random.randint(0, high=len(games))
    #print(game_index)
    if not os.path.exists(os.path.join(IMG_DIR, game['image']['file_name'])):
        img_fname = wget.download(game['image']['flickr_url'])
        os.rename(img_fname, os.path.join(IMG_DIR, game['image']['file_name']))
    img = Image.open(os.path.join(IMG_DIR, game['image']['file_name']))
    osize = (game['image']['width'], game['image']['height'])
    img = img.resize((osize), resample=Image.NEAREST)

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
            print(i, obj['category'], '<--- target')
        else:
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
            ax.text(x_right, y_upper-1, str(i), color='g', fontsize=34)
        else:
            ax.text(x_right, y_upper-1, str(i), color='r', fontsize=34)

    plt.show()
