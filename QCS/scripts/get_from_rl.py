import json

from gw_utils import get_raw_dataset


if __name__ == "__main__":

    dataset = get_raw_dataset('data/guesswhat.test.jsonl.gz')

    game_keys = {}
    for game in dataset:
        entry = {'image_id': str(game['image']['id']),
                 'target_object_id': str(game['object_id'])}
        game_keys[str(game['id'])] = entry

    rawtext = open('data/Trento_GuessWhat_dialogue_AAAI20_2W.txt', 'r')
    
    data = {}
    line = rawtext.readline()
    line = rawtext.readline()

    for i, line in enumerate(rawtext):
        entry_id = None
        ids, qas, success, target_object_id, guessed_object_id = line.split('\t')
        qas = qas.replace('||', ' ').replace('#', ' ')

        # As every game is related to a pair (image_id, target_object_id)
        # this will always match an entry in rawtext with an entry in 
        # game_keys

        for k in game_keys.keys():
            entry = game_keys[k]
            #print(type(target_object_id), type(entry['target_object_id']))
            same_target = target_object_id == entry['target_object_id']
            #print(type(ids), type(entry['image_id']))
            same_img = ids == entry['image_id']
            if same_target and same_img:
                entry_id = k
                game_keys.pop(k)
                break

        if entry_id == None:
            print(ids + ' + ' + target_object_id + ' not matched')

        data[entry_id] = {'image_id': ids, 'gen_dialogue': qas,
                          'state': 'success' if success=='True' else 'failure',
                          'target_object_id': target_object_id,
                          'guessed_object_id': guessed_object_id[:-1]}
        #data.append({'image_id': ids, 'gen_dialogue': qas,
        #             'state': 'success' if success=='True' else 'failure',
        #             'target_object_id': target_object_id,
        #             'guessed_object_id': guessed_object_id[:-1]})

        #if i % 100 == 0:
        #    print(str(i) + ' done')

    with open('aaai20_dialogues.json', 'w') as fl:
        json.dump(data, fl)

    #rawtext = open('data/Trento_GuessWhat_dialogue_CVPR20_2W.txt', 'r')

    #data = []

    #for line in rawtext:
    #    ids, qas, success = line.split('\t')
    #    qas = qas.replace('||', ' ').replace('#', ' ')

    ##data[gwdata[ids]] = {'gen_dialogue': qas,
    #data.append({'image_id': ids, 'gen_dialogue': qas,
    #             'state': 'success' if success else 'failure'})

    #with open('cvpr20_dialogues.json', 'w') as fl:
        #json.dump(data, fl)

