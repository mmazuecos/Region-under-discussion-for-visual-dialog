import os
import json
import h5py
import gzip
import io
import copy
import torch
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from numpy.testing import assert_almost_equal
from nltk.tokenize import TweetTokenizer
from utils.image_utils import get_spatial_feat
from utils.get_more_history import tagged_parse
from torch.utils.data import Dataset
from nltk.stem import WordNetLemmatizer

import numpy as np
import os
#from PIL import Image
#from torchvision import transforms
#from models.CNN import ResNet


def get_distractors(game, hist, cats, supercats):
    """
    Compute distractors. Returns a list of distractor ids.
    May not include the target id.

    game -- game dictionary
    hist -- complete history (non-empty list of pairs (relation, referent))
    cats -- COCO categories
    supercats -- map of super-categories to categories
    """
    assert hist  # history is not empty

    objs = game['objects']

    # NEW APPROACH
    """pos_cats, neg_cats = set(cats), set()
    for rel, ref in hist:
        if rel.startswith('NOT_'):
            if ref in cats:
                neg_cats.add(ref)
            else:
                # ref in super-categories
                neg_cats.update(supercats[ref])
        else:
            # positive history
            if ref in cats:
                pos_cats.intersection_update({ref})
            else:
                # ref in super-categories
                pos_cats.intersection_update(supercats[ref])

    final_cats = pos_cats - neg_cats
    if not final_cats:
        print(f"WARNING: contradictory history: {hist} - {objs[0]['id']}")

    dist_objs = [o for o in objs if o['category'] in final_cats]"""

    # first discard objects using negative history
    neg_cats = []
    for rel, ref in hist:
        if rel.startswith('NOT_'):
            if ref in cats:
                neg_cats.append(ref)
            else:
                # ref in super-categories
                neg_cats.extend(supercats[ref])
    dist_objs = [o for o in objs if o['category'] not in neg_cats]

    # now get last positive history, if present, and use it
    # TODO: study other strategies here
    pos_hist = [h for h in hist if h[0].startswith('is_')]
    if pos_hist:
        rel, ref = pos_hist[-1]
        if ref in cats:
            pos_cats = [ref]
        else:
            # ref in super-categories
            pos_cats = supercats[ref]
        final_cats = list(set(pos_cats) - set(neg_cats))
        if not final_cats:
            print(f"WARNING: contradictory history: {hist} - game {game['id']}")
        dist_objs = [o for o in dist_objs if o['category'] in pos_cats]

    # return ids
    distractors = [o['id'] for o in dist_objs]
    return distractors


def constrain_bbox(game, distractors):
    """
    Compute focused spatial information.
    The target is always included.
    If there are no distractors, the focus is only on the target.

    game -- json for the game
    distractors -- list of object ids
    """
    objs = game['objects']
    target_id = game['object_id']
    target = [o for o in objs if o['id'] == target_id][0]
    target_bbox = target['bbox']

    if not distractors:
        print(f"WARNING: empty distractors in game {game['id']}.")

    focus_ids = list(distractors)
    if target_id not in focus_ids:
        # always add target
        #print(f"WARNING: target not a distractor in game {game['id']}.")
        focus_ids.append(target_id)

    focus = [o for o in objs if o['id'] in focus_ids]

    objects_bbox = [np.inf, np.inf, -np.inf, -np.inf]
    for o in focus:
        # Get object bbox
        bbox = o['bbox']
        # read left, upper, widht, height
        # generate left, upper, right, lower
        objects_bbox[0] = min(objects_bbox[0], bbox[0])
        objects_bbox[1] = min(objects_bbox[1], bbox[1])
        objects_bbox[2] = max(objects_bbox[2], bbox[0] + bbox[2])
        objects_bbox[3] = max(objects_bbox[3], bbox[1] + bbox[3])

    # go back to left, upper, widht, height
    obwidth = objects_bbox[2] - objects_bbox[0]
    obheight = objects_bbox[3] - objects_bbox[1]
    objects_bbox[2] = obwidth
    objects_bbox[3] = obheight

    # Shift bounding boxes to the new size
    target_bbox = list(target_bbox)
    target_bbox[0] -= objects_bbox[0]
    target_bbox[1] -= objects_bbox[1]

    # Get spatial information
    spatial = get_spatial_feat(bbox=target_bbox, im_width=obwidth, im_height=obheight)
    return spatial, objects_bbox


def check_spatial(spatial):
    """Check for spatial information consistency.

    spatial -- list of the form
      [x_min, y_min, x_max, y_max, x_center, y_center, w_box, h_box]
    """
    top = np.ones(6)
    bottom = np.array([-1.] * 6)
    assert_almost_equal(np.maximum(top, spatial[:6]), top)
    assert_almost_equal(np.minimum(bottom, spatial[:6]), bottom)

    x_min, y_min, x_max, y_max, x_center, y_center, w_box, h_box = spatial
    assert_almost_equal(x_max - x_min, w_box)
    assert_almost_equal(y_max - y_min, h_box)
    assert_almost_equal(x_center, (x_min + x_max) / 2.)
    assert_almost_equal(y_center, (y_min + y_max) / 2.)


lemmatizer = WordNetLemmatizer()

def semantic_parse(qa, hist, cats, supercats, lemmatize=False, negative=False, second=False):
    """Parse question and return a new history element.

    qa -- question dictionary (need keys 'tagged' and 'answer')
    hist -- previous history (list of tuples)
    cats -- COCO categories
    supercats -- map of super-categories to categories
    lemmatize -- whether to lemmatize relation referent
    negative -- whether to include negative history ('No' answers)
    second -- whether to try second token for two-token positive history
    """
    # TODO: do something with duplicate history?
    # TODO: add object list in scene as parameter. use it.

    rel_types = {'is_a', 'is_the'}
    if negative:
        rel_types.update({'NOT_is_a', 'NOT_is_the'})

    rel_tup = tagged_parse(qa)
    if rel_tup is not None and rel_tup[0] in rel_types:
        # process rel_tup
        rel, ref = rel_tup
        sref = ref.split()
        rwords, rtags = zip(*[r.split('_') for r in sref])
        if lemmatize:
            rwords = [lemmatizer.lemmatize(w) for w in rwords]
        ref = '_'.join(rwords)
        tag = '_'.join(rtags)
        #tags[tag] += 1

        # Use if referent is a category or a super-category
        use = ref in cats or ref in supercats
        if second and not use and len(rwords) == 2 and rel.startswith('is_'):
            # Second chance for two-token positive history:
            # Try with second token only
            ref = rwords[1]
            use = ref in cats or ref in supercats

        # check if history contradicts previous elements
        if use:
            if rel.startswith('is_'):
                neg_rel = 'NOT_' + rel
            else:
                assert rel.startswith('NOT_is_')
                neg_rel = rel[4:]
            if (neg_rel, ref) in hist:
                use = False
                print(f"WARNING: skipped contradictory history item: {(rel, ref)}")

        if use:
            return (rel, ref)
        else:
            return None
    else:
        return None


class OracleDataset(Dataset):
    def __init__(self, data_dir, data_file, split, visual_feat_file, visual_feat_mapping_file,
                 visual_feat_crop_file, visual_feat_crop_mapping_file, max_src_length, hdf5_visual_feat,
                 hdf5_crop_feat, history = False, new_oracle_data=False,
                 successful_only=True, min_occ=3, load_crops=False,
                 negative=False, supercats=False, second=False):
        self.data_dir = data_dir
        self.split = split
        self.parsed_fname = os.path.join(data_dir, 'guesswhat.'+self.split+'.parsed.json')
        self.successful_only = successful_only
        # This "history" parameter corresponds to the raw text history
        # concatenated at the beggining of the dialogue.
        self.history = history
        # where to save/load preprocessed data
        self.data_file_name = 'oracle_' + split
        if self.history:
            self.data_file_name += '_history'
        if not self.successful_only:
            self.data_file_name += '_all'
        self.data_file_name += '_data.json'
        #if self.history:
        #    self.data_file_name += 'oracle_' + split + '_history_data.json'
        #else:
        #    self.data_file_name = 'oracle_' + split + '_data.json'
        self.vocab_file_name = 'vocab.json'
        self.mscoco_cats_file = 'mscoco_cats'
        self.supercats_file = 'supercategories_to_categories.json'
        self.min_occ = min_occ
        # original guesswhat data
        self.data_file = data_file
        self.visual_feat_file = os.path.join(data_dir, visual_feat_file)
        self.visual_feat_crop_file = os.path.join(data_dir, visual_feat_crop_file)
        self.max_src_length = max_src_length
        self.hdf5_visual_feat = hdf5_visual_feat
        self.hdf5_crop_feat = hdf5_crop_feat
        self.load_crops = load_crops
        self.negative = negative
        self.supercats = supercats
        self.second = second

        if self.history:
            self.max_diag_len = self.max_src_length*2+1
        else:
            self.max_diag_len = None

        # Load vfeats
        self.vf = np.asarray(h5py.File(self.visual_feat_file, 'r')[self.hdf5_visual_feat])
        if self.load_crops:
            # Load crop feats
            self.cf = np.asarray(h5py.File(self.visual_feat_crop_file, 'r')[self.hdf5_crop_feat])
            assert(self.cf.dtype == np.float32)

            # Load crop feats mapping
            with open(os.path.join(data_dir, visual_feat_crop_mapping_file), 'r') as file_c:
                self.visual_feat_crop_mapping_file = json.load(file_c)
            #self.visual_feat_crop_mapping_file = self.visual_feat_crop_mapping_file['crops_features2id']

        # Load vfeats mapping
        with open(os.path.join(data_dir, visual_feat_mapping_file), 'r') as file_v:
            self.visual_feat_mapping_file = json.load(file_v)


        #self.visual_feat_mapping_file = self.visual_feat_mapping_file[split+'2id']

        # load or create new vocab
        with open(os.path.join(data_dir, self.vocab_file_name), 'r') as file:
            self.word2i = json.load(file)['word2i']

        with open(os.path.join(data_dir, self.mscoco_cats_file), 'r') as file:
            self.mscoco_cats = [word[:-1].replace(' ', '_') for word in file]

        if supercats:
            with open(os.path.join(data_dir, self.supercats_file), 'r') as file:
                super2cat = json.load(file)
            for k in super2cat:
                cats = super2cat[k]
                super2cat[k] = [cat.replace(' ', '_') for cat in cats]
            self.super2cat = super2cat
        else:
            self.super2cat = {}

        # create new oracle_data file or load from disk
        if new_oracle_data or not os.path.isfile(os.path.join(self.data_dir, self.data_file_name)):
            self.oracle_data = self.new_oracle_data()
        else:
            print("Loading " + self.data_file_name + " File.")
            with open(os.path.join(self.data_dir, self.data_file_name), 'r') as file:
                self.oracle_data = json.load(file)

    def __len__(self):
        return len(self.oracle_data)

    def __getitem__(self, idx):

        if not type(idx) == str:
            idx = str(idx)

        # load image features
        #visual_feat_id = self.visual_feat_mapping_file[self.oracle_data[idx]['image_file']]
        #visual_feat = self.vf[visual_feat_id]
        visual_feat = 0
        if self.load_crops:
            game_id = self.oracle_data[idx]['game_id']
            crop_feat_id = self.visual_feat_crop_mapping_file[game_id]
            crop_feat = self.cf[crop_feat_id]
        else:
            crop_feat = 0
        
        question = np.asarray(self.oracle_data[idx]['question'])
        lenght = self.oracle_data[idx]['length']

        # Define output tuple
        output = {'question': question,
                  'answer': self.oracle_data[idx]['answer'],
                  'crop_features': crop_feat,
                  'img_features': visual_feat,
                  'spatial': np.asarray(self.oracle_data[idx]['spatial'], dtype=np.float32),
                  'focus': np.asarray(self.oracle_data[idx]['focus'], dtype=np.float32),
                  'obj_cat': self.oracle_data[idx]['obj_cat'],
                  'length': lenght,
                  'game_id': self.oracle_data[idx]['game_id'],
                  'hist': self.oracle_data[idx]['hist']}

        return output

    def new_oracle_data(self):

        print("Creating New " + self.data_file_name + " File.")

        path = os.path.join(self.data_dir, self.data_file)
        tknzr = TweetTokenizer(preserve_case=False)
        oracle_data = dict()
        _id = 0
        _img_id = 0

        ans2tok = {'Yes': 1,
                   'No': 0,
                   'N/A': 2}

        with open(self.parsed_fname) as fl:
            parses = json.load(fl)

        with gzip.open(path) as file:
            for json_game in file:
                hist = []
                new_spatial = None
                focus_bbox = None
                distractors = None
                hist_updated = False
                game = json.loads(json_game.decode("utf-8"))

                if self.successful_only:
                    if not game['status'] == 'success':
                        continue

                if self.history:
                    prev_ques = list()
                    prev_answer = list()
                    prev_length = 0

                # Get target object and distractors info
                objs_by_cat = {}
                for i, o in enumerate(game['objects']):
                    if not (o['category'] in objs_by_cat.keys()):
                        objs_by_cat[o['category']] = [o]
                    else:
                        objs_by_cat[o['category']].append(o)

                    if o['id'] == game['object_id']:
                        # save target object information
                        target = o

                # spatial information
                spatial = get_spatial_feat(
                    bbox=target['bbox'],
                    im_width=game['image']['width'],
                    im_height=game['image']['height'])
                check_spatial(spatial)

                for i, qa in enumerate(game['qas']):
                    q_tokens = tknzr.tokenize(qa['question'])
                    q_token_ids = [self.word2i[w] if w in self.word2i else self.word2i['<unk>'] for w in q_tokens][:self.max_src_length]
                    a_token = ans2tok[qa['answer']]

                    length = len(q_token_ids)

                    # Question encoding
                    if self.history:
                        question = prev_ques+prev_answer+q_token_ids
                        question_length = prev_length+length
                    else:
                        question = q_token_ids
                        question_length = length

                    # Question padding
                    if self.history:
                        question.extend([self.word2i['<padding>']] * (self.max_diag_len - len(question)))
                    else:
                        question.extend([self.word2i['<padding>']] * (self.max_src_length - len(question)))

                    # Constrain the region
                    if hist != [] and hist_updated:
                        distractors = get_distractors(game, hist, self.mscoco_cats, self.super2cat)
                        new_spatial, focus_bbox = constrain_bbox(game, distractors)
                        if new_spatial is not None:
                            check_spatial(new_spatial)
                        hist_updated = False

                    assert _id not in oracle_data
                    oracle_data[_id] = {
                        'question': question,
                        'length': question_length,
                        'answer': a_token,
                        'image_file': game['image']['file_name'],
                        # Modify this in order to always load the correct spatial
                        # feature
                        'spatial': spatial,
                        'focus': new_spatial if new_spatial is not None else spatial,
                        'focus_bbox': focus_bbox,
                        'dist': distractors,
                        'target_id': target['id'],
                        'game_id': str(game['id']),
                        'obj_cat': target['category_id'],
                        'vfeats_id': _img_id,
                        'hist': copy.deepcopy(hist),
                    }

                    prev_ques = copy.deepcopy(q_token_ids)
                    prev_answer = [copy.deepcopy(a_token)]
                    prev_length = length+1

                    _id += 1

                    # Extract relation and referent of the relation 
                    qa['tagged'] = parses[str(qa['id'])]
                    rel_tup = semantic_parse(qa, hist, self.mscoco_cats,
                                             self.super2cat, lemmatize=True,
                                             negative=self.negative,
                                             second=self.second)
                    if rel_tup is not None:
                        hist.append(rel_tup)
                        hist_updated = True
                        _img_id += 1

        oracle_data_path = os.path.join(self.data_dir, self.data_file_name)
        with io.open(oracle_data_path, 'wb') as f_out:
            data = json.dumps(oracle_data, ensure_ascii=False)
            f_out.write(data.encode('utf8', 'replace'))

        print('done')

        with open(oracle_data_path, 'r') as file:
            oracle_data = json.load(file)

        return oracle_data

    def print_statistics(self):
        # history statistics
        hcounts = Counter()
        for q in self.oracle_data.values():
            for h in q['hist']:
                rel = 'yes' if h[0].startswith('is_') else 'no'
                ref = h[1]
                cat = 'cat' if ref in self.mscoco_cats else 'sup'
                hcounts[rel, ref, cat] += 1
        n = 20
        print(f'Top {n} history entries:')
        for (rel, ref, cat), count in hcounts.most_common(n):
            print(f'{rel}\t{ref:10}\t{cat}\t{count:6}')
        print()

        # distractor statistics
        dcounts = defaultdict(int)
        for q in self.oracle_data.values():
            if q['dist'] is not None:
                #has_dist = len(q['dist']) > 0
                #tgt_in_dist = q['target_id'] in q['dist']
                #positive = not q['hist'][-1][0].startswith('NOT_')
                #counts[has_dist, tgt_in_dist, positive] += 1
                if len(q['dist']) == 0:
                    dcounts['empty'] += 1
                elif q['target_id'] not in q['dist']:
                    dcounts['no_tgt'] += 1
                elif q['dist'] == [q['target_id']]:
                    dcounts['only_tgt'] += 1
                else:
                    dcounts['more'] += 1
            else:
                dcounts['no'] += 1

        print(f"No distractors: {dcounts['no']}")
        print(f"Empty list of distractors: {dcounts['empty']}")
        print(f"Non-empty, target not present: {dcounts['no_tgt']}")
        print(f"Only target: {dcounts['only_tgt']}")
        print(f"Target and more distractors: {dcounts['more']}")
