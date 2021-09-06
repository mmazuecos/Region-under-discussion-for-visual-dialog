import re
from collections import defaultdict
import json
import jsonlines
from tqdm import tqdm

from gw_utils import get_raw_dataset

# https://docs.python.org/3/howto/regex.html#regex-howto
# https://docs.python.org/3/library/re.html
pref = r'(?:^|is_VBZ it_PRP |it_PRP is_VBZ |is_VBZ (?:the|this)_DT \w+_NN |is_VBZ s?he_PRP |are_VBP they_PRP )'
suf = r'(?:[\?\.,_\s])*$'  # some spaces ? , . _ followed by end of string


rules = [
    ('is_a',   pref + r'(?:an?_DT|one_CD of_IN the_DT) (?P<val>\w+)_\w+' + suf),
    ('is_the', pref + r'the_DT (?P<val>\w+)_NN(?:S|P|PS)?' + suf),
    ('in_a',   pref + r'(?:the_DT one_NN )?in_IN (an?_DT|one_CD of_IN the_DT) (?P<val>\w+)_\w+' + suf),
    ('in_the', pref + r'(?:the_DT one_NN )?in_IN the_DT (?P<val>\w+)_\w+' + suf),
    ('on_a',   pref + r'(?:the_DT one_NN )?on_IN (an?_DT|one_CD of_IN the_DT) (?P<val>\w+)_\w+' + suf),
    ('on_the', pref + r'(?:the_DT one_NN )?on_IN the_DT (?P<val>\w+)_\w+' + suf),
    ('attr',   pref + r'(?P<val>\w+)_JJ' + suf),

    # special rules for "is a/the NN" ('this' and 'that' cuold also work)
    ('is_a',   r'^is_VBZ an?_DT (?P<val>\w+)_NN' + suf),
    ('is_the', r'^is_VBZ the_DT (?P<val>\w+)_NN' + suf),

    # XXX: matches "is_VBZ the_DT table_NN cloth_NN ?_." as ('is_a', 'cloth')
    ('is_a',   pref + r'(?P<val>\w+)_NN(?:S|P|PS)?' + suf),

    # compound nouns: two tokens NN NN
    ('is_a',   pref + r'(?:an?_DT|one_CD of_IN the_DT) (?P<val1>\w+)_NN(?:S|P|PS)? (?P<val2>\w+)_NN(?:S|P|PS)?' + suf),
    ('is_the', pref + r'the_DT (?P<val1>\w+)_NN(?:S|P|PS)? (?P<val2>\w+)_NN(?:S|P|PS)?' + suf),
    ('in_a',   pref + r'(?:the_DT one_NN )?in_IN (?:an?_DT|one_CD of_IN the_DT) (?P<val1>\w+)_NN(?:S|P|PS)? (?P<val2>\w+)_NN(?:S|P|PS)?' + suf),
    ('in_the', pref + r'(?:the_DT one_NN )?in_IN the_DT (?P<val1>\w+)_NN(?:S|P|PS)? (?P<val2>\w+)_NN(?:S|P|PS)?' + suf),
    ('on_a',   pref + r'(?:the_DT one_NN )?on_IN (?:an?_DT|one_CD of_IN the_DT) (?P<val1>\w+)_NN(?:S|P|PS)? (?P<val2>\w+)_NN(?:S|P|PS)?' + suf),
    ('on_the', pref + r'(?:the_DT one_NN )?on_IN the_DT (?P<val1>\w+)_NN(?:S|P|PS)? (?P<val2>\w+)_NN(?:S|P|PS)?' + suf),
    ('is_a',   pref + r'(?P<val1>\w+)_NN(?:S|P|PS)? (?P<val2>\w+)_NN(?:S|P|PS)?' + suf),
 
    # noun with adjective: JJ NN
    ('is_a',   pref + r'(?:an?_DT|one_CD of_IN the_DT) (?P<val1>\w+)_JJ (?P<val2>\w+)_NN(?:S|P|PS)?' + suf),
    ('is_the', pref + r'the_DT (?P<val1>\w+)_JJ (?P<val2>\w+)_NN(?:S|P|PS)?' + suf),
    ('in_a',   pref + r'(?:the_DT one_NN )?in_IN (?:an?_DT|one_CD of_IN the_DT) (?P<val1>\w+)_JJ (?P<val2>\w+)_NN(?:S|P|PS)?' + suf),
    ('in_the', pref + r'(?:the_DT one_NN )?in_IN the_DT (?P<val1>\w+)_JJ (?P<val2>\w+)_NN(?:S|P|PS)?' + suf),
    ('on_a',   pref + r'(?:the_DT one_NN )?on_IN (?:an?_DT|one_CD of_IN the_DT) (?P<val1>\w+)_JJ (?P<val2>\w+)_NN(?:S|P|PS)?' + suf),
    ('on_the', pref + r'(?:the_DT one_NN )?on_IN the_DT (?P<val1>\w+)_JJ (?P<val2>\w+)_NN(?:S|P|PS)?' + suf),
    ('is_a',   pref + r'(?P<val1>\w+)_JJ (?P<val2>\w+)_NN(?:S|P|PS)?' + suf),

    # have
    ('have', r'does_VBZ (?:it_PRP|(?:the|this)_DT \w+_NN|s?he_PRP) have_VB (?:an?_DT )?(?P<val>\w+)_\w+' + suf),
    ('have', r'do_VBP they_PRP have_VB (?:an?_DT )?(?P<val>\w+)_\w+' + suf),
    
    # attr: "does it fly?"
    ('attr', r'does_VBZ (?:it_PRP|(?:the|this)_DT \w+_NN|s?he_PRP) (?P<val>\w+)_\w+' + suf),
]

rules_regex = [(rel, re.compile(regex, flags=re.IGNORECASE)) for (rel, regex) in rules]


def parse(q):
    """Rule-based semantic parse for question q.
    If a rule matches, returns a pair (rel, ref) where:
      - rel is the relation type
      - ref is the referent (tokens joined with '_')
    """
    if q['answer'] == 'N/A':
        return None
    for rel, regex in rules_regex:
        match = regex.match(q['tagged'])
        if match:
            if q['answer'] == 'No':
                rel = 'NOT_' + rel
            return (rel, '_'.join(match.groups()).lower())
    return None


tagged_rules = [
    ('is_a',   pref + r'(?:an?_DT|one_CD of_IN the_DT) (?P<val>\w+_\w+)' + suf),
    ('is_the', pref + r'the_DT (?P<val>\w+_NN(?:S|P|PS)?)' + suf),

    # special rules for "is a/the NN" ('this' and 'that' cuold also work)
    ('is_a',   r'^is_VBZ an?_DT (?P<val>\w+_NN)' + suf),
    ('is_the', r'^is_VBZ the_DT (?P<val>\w+_NN)' + suf),

    # XXX: matches "is_VBZ the_DT table_NN cloth_NN ?_." as ('is_a', 'cloth')
    ('is_a',   pref + r'(?P<val>\w+_NN(?:S|P|PS)?)' + suf),

    # compound nouns: two tokens NN NN
    ('is_a',   pref + r'(?:an?_DT|one_CD of_IN the_DT) (?P<val1>\w+_NN(?:S|P|PS)?) (?P<val2>\w+_NN(?:S|P|PS)?)' + suf),
    ('is_the', pref + r'the_DT (?P<val1>\w+_NN(?:S|P|PS)?) (?P<val2>\w+_NN(?:S|P|PS)?)' + suf),
    ('is_a',   pref + r'(?P<val1>\w+_NN(?:S|P|PS)?) (?P<val2>\w+_NN(?:S|P|PS)?)' + suf),

    # noun with adjective: JJ NN
    ('is_a',   pref + '(?:an?_DT|one_CD of_IN the_DT) (?P<val1>\w+_JJ) (?P<val2>\w+_NN(?:S|P|PS)?)' + suf),
    ('is_the', pref + 'the_DT (?P<val1>\w+_JJ) (?P<val2>\w+_NN(?:S|P|PS)?)' + suf),
    ('is_a',   pref + r'(?P<val1>\w+_JJ) (?P<val2>\w+_NN(?:S|P|PS)?)' + suf),
]

tagged_rules_regex = [(rel, re.compile(regex, flags=re.IGNORECASE)) for (rel, regex) in tagged_rules]


def tagged_parse(q):
    """Rule-based semantic parse for question q.
    If a rule matches, returns a pair (rel, ref) where:
      - rel is the relation type
      - ref is the referent (tagged tokens 'w_T' joined with ' ')
    """
    if q['answer'] == 'N/A':
        return None
    for rel, regex in tagged_rules_regex:
        match = regex.match(q['tagged'])
        if match:
            if q['answer'] == 'No':
                rel = 'NOT_' + rel
            return (rel, ' '.join(match.groups()).lower())
    return None


if __name__ == "__main__":
    datasets = [
        'data/guesswhat.train.jsonl.gz',
        'data/guesswhat.valid.jsonl.gz',
        'data/guesswhat.test.jsonl.gz',
        #'data/compguesswhat.valid.jsonl.gz',
    ]
    parses = [
        'data/guesswhat.train.parsed.json',
        'data/guesswhat.valid.parsed.json',
        'data/guesswhat.test.parsed.jsonl',
        #'data/compguesswhat.valid.parsed.json'
    ]

    for ds, p in zip(datasets, parses):
        print('Dataset:', ds)
        with open(p) as f:
            parses = json.load(f)

        games = get_raw_dataset(ds)

        histories = []
        total = 0
        #for _, game in zip(range(100), games):
        for game in tqdm(games):
            history = []
            for i, q in enumerate(game['qas']):
                q['tagged'] = parses[str(q['id'])]
                rel = parse(q)
                if rel:
                    history.append((i,) + rel)

            histories.append(history)
            total += len(game['qas'])

        output_fn = ds[:-9] + '.more_history.jsonl'
        with jsonlines.open(output_fn, 'w') as f:
            f.write_all(histories)

        found = sum(len(h) for h in histories)
        print(f'Found {found} relation tuples out of {total} questions.')

        # stats
        stats = defaultdict(lambda: defaultdict(list))
        for h in histories:
            for id, rel, token in h:
                if rel.startswith('NOT_'):
                    rel = rel[4:]
                    stats[rel]['No'].append(token)
                else:
                    stats[rel]['Yes'].append(token)

        # print stats
        print('\tYes\tNo')
        for rel, rstats in stats.items():
            print(f"{rel}\t{len(rstats['Yes']):4}\t{len(rstats['No']):4}")
