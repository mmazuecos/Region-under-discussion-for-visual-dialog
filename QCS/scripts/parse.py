import json
import stanza
from gw_utils import get_raw_dataset
from tqdm import tqdm


def parse(nlp, games, batch_size=100):
    # COLLECT QUESTIONS
    questions = []
    #for _, game in zip(range(5), games):
    for game in games:
        for q in game['qas']:
            questions.append((q['id'], q['question']))

    # DIVIDE IN BATCHES AND PARSE:
    parses = []
    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    for i in tqdm(range(0, len(questions), batch_size)):
        qbatch = questions[i:i + batch_size]
        text = ''
        for id, qs in qbatch:
            text += qs + '\n\n'

        doc = nlp(text)

        for sent in doc.sentences:
            assert len(sent.tokens) == len(sent.words)
            parse = ' '.join(word.text + '_' + word.xpos for word in sent.words)
            parses.append(parse)

    assert len(questions) == len(parses)
    parses_dict = {}
    for (id, q), p in zip(questions, parses):
        parses_dict[id] = p
    
    return parses_dict


if __name__ == "__main__":
    nlp = stanza.Pipeline('en', processors='tokenize,pos', tokenize_no_ssplit=True)

    datasets = [
        'data/guesswhat.train.jsonl.gz',
        'data/guesswhat.valid.jsonl.gz',
        'data/guesswhat.test.jsonl.gz',
        #'data/compguesswhat.valid.jsonl.gz'
    ]

    for ds in datasets:
        games = get_raw_dataset(ds)

        parses_dict = parse(nlp, games, batch_size=100)  # batch_size in number of questions

        output_fn = ds[:-9] + '.parsed.json'
        with open(output_fn, 'w') as f:
            json.dump(parses_dict, f, sort_keys=True, indent="")
