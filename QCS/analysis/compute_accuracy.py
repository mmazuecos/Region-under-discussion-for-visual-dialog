import json
from nltk.tokenize import TweetTokenizer
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from analysis.qclassify import qclass
classifier = qclass()

def accuracy_qtype(games, numq, hist):
    """
    Compute accuracy by question type.
    """
    print('Computing accuracy by question type...')

    tknzr = TweetTokenizer(preserve_case=False)
    q_count = 0

    qsby_qtype = {'all': [],
                  'object': [],
                  'super-category': [],
                  'color': [],
                  'shape': [],
                  'size': [],
                  'texture': [],
                  'action': [],
                  'spatial': [],
                  'N/A': []}

    for key in games:
        # Ignore non successful games
        if not games[key]['state'] == 'success':
            continue
        questions = games[key]['qas']
        for que_idx, que in enumerate(questions):
            if hist == 1 and que['hist'] == []:
                continue
            elif hist == 0 and que['hist'] != []:
                continue

            q_count +=1
            cat = '<NA>'
            cat = classifier.que_classify_multi(que['question'].lower())

            att_flag = False

            qsby_qtype['all'].append(que)
            if '<color>' in cat:
                qsby_qtype['color'].append(que)
            if '<shape>' in cat:
                qsby_qtype['shape'].append(que)
            if '<size>' in cat:
                qsby_qtype['size'].append(que)
            if '<texture>' in cat:
                qsby_qtype['texture'].append(que)
            if '<action>' in cat:
                qsby_qtype['action'].append(que)
            if '<spatial>' in cat:
                qsby_qtype['spatial'].append(que)
            if '<object>' in cat:
                qsby_qtype['object'].append(que)
            if '<super-category>' in cat:
                qsby_qtype['super-category'].append(que)
            if cat == '<NA>':
                qsby_qtype['N/A'].append(que)

    total_correct = 0
    total_qs = 0

    if numq:
        for qtype in qsby_qtype:
            ques = qsby_qtype[qtype]
            qtype_count = len(ques)
            correct = sum([q['ans'] == q['model_ans'] for q in ques])
            print('{}: {:.2f}, Support: {}'.format(qtype, correct/qtype_count*100, qtype_count))
            total_correct += correct
            total_qs += qtype_count
    else:
        for qtype in qsby_qtype:
            ques = qsby_qtype[qtype]
            qtype_count = len(ques)
            correct = sum([q['ans'] == q['model_ans'] for q in ques])
            print('{}: {:.2f}'.format(qtype, correct/qtype_count*100))
            total_correct += correct
            total_qs += qtype_count


def align_data(games, qclassification):
    """
    Auxiliar function to align the Q&As data to the location classification.
    """
    mudata = {'game_id':[],
      'dial_pos':[],
      'model_answer':[],
      'hist': []}

    for k in games:
        for i, qa in enumerate(games[k]['qas']):
            mudata['game_id'].append(k)
            mudata['dial_pos'].append(i)
            mudata['model_answer'].append(qa['model_ans'])
            mudata['hist'].append(qa['hist'])

    mudata = pd.DataFrame(mudata)
    locdata = qclassification.merge(mudata, on=['game_id', 'dial_pos'], how='left') 
    return locdata 


def filter_whist(locdata, hist):
    """
    Auxiliar function to filter for questions with or without history.
    """
    if hist == -1:
        return locdata
    else:
        locdata['hist'] = locdata['hist'].astype(str)
        whist = locdata['hist'] != '[]'
        if hist == 1:
            new_locdata = locdata[whist]
        else:
            new_locdata = locdata[~whist]
        return new_locdata 


def accuracy_loctype(games, numq, hist):
    """
    Compute accuracy by location question type.
    """
    print('Computing accuracy for location questions...')
    # Hardcoded path. Sry
    qclassification = pd.read_csv('./data/val-q_classification-successful-only.csv')
    qclassification['game_id'] = qclassification['game_id'].astype(str)

    locdata = align_data(games, qclassification)
    # Change this for the relevant data
    locdata = filter_whist(locdata, hist)

    absolute = locdata[locdata['location_type'] == 'absolute']
    true_absolute = absolute[absolute['is_true_absolute']]
    crap_absolute = absolute[~absolute['is_true_absolute']]

    relational = locdata[locdata['location_type'] == 'relational']
    true_relational = relational[relational['is_true_relational']]
    crap_relational = relational[~relational['is_true_relational']]

    number = locdata[locdata['location_type'] == 'number']
    
    crap = pd.concat([crap_absolute, crap_relational])

    data = zip(['absolute', 'relational', 'number', 'other'],
               [true_absolute, true_relational, number, crap])

    if numq:
        for loctype, locdata in data: 
            count = (locdata['answer'] == locdata['model_answer']).value_counts()
            accuracy = 100 * count.values[0] / len(locdata)
            print('{}: {:.2f} Support: {}'.format(loctype, accuracy, len(locdata)))
            #print('{}'.format(accuracy))
    else:
        for loctype, locdata in data: 
            count = (locdata['answer'] == locdata['model_answer']).value_counts()
            accuracy = 100 * count.values[0] / len(locdata)
            print('{}: {:.2f}'.format(loctype, accuracy))
            #print('{}'.format(accuracy))

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='File of the games and the answers from the model.')
    parser.add_argument('--hist', type=int, default=-1, help='Compute accuracy for games with or without history. Options: 0 for no history, 1 for history, -1 for all the games. Default: -1.')
    parser.add_argument('--numq', default=False, action='store_true', help='Show the support for each type of question.')
    parser.add_argument('--noqtype', default=False, action='store_true', help='Do not report accurary by question type.')
    parser.add_argument('--noloctype', default=False, action='store_true', help='Do not report accuracy by location question type.') 
    #TODO
    #parser.add_argument('-rawout', default=False, action='store_true', help='Print only the plain numbers.')
    #parser.add_argument('-save', default=False, action='store_true', help='Save results to a csv.')

    args = parser.parse_args()

    assert(args.hist == -1 or args.hist == 0 or args.hist == 1)

    print('Loading '+ args.file+ '...')
    with open(args.file) as fl:
        games = json.load(fl)

    if not args.noqtype:
        accuracy_qtype(games, args.numq, args.hist)
    if not args.noloctype:
        accuracy_loctype(games, args.numq, args.hist)

