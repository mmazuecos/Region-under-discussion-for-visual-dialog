import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import math

def pos(idx):
    return math.floor(idx/2)

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument("-prefix", type=str, help='Prefix of the files for effectiveness logs in the GuessWhat?! dataset.')
    parser.add_argument("-to_file", action='store_true', help='Dump summary to file.')
    parser.add_argument("-out_fname", type=str, default='out.json', help='Name of the summary file if -to_file is set.')
    parser.add_argument("-lenght", type=int, default=8, help='Amount of questions to process.')

    args = parser.parse_args()

    dial_fname = 'data/'+args.prefix+'_dialogues.json'
    eff_fname = 'data/'+args.prefix+'_effectiveness.json'

    print(dial_fname)
    print(eff_fname)
 
    with open(eff_fname, 'r') as fl:
        dials = json.load(fl)

    with open(dial_fname, 'r') as fl:
        states = json.load(fl)
    
    # Data to get from file 
    total_effective = 0
    total_questions = 0
    per_dial_effective_avg = 0
    fail_avg_eff = 0
    succ_avg_eff = 0
    total_success = 0
    total_fail = 0
    total_dial = 0
    # TODO make this not hardcoded
    rm_target_dial = 0
    if args.prefix != 'human':
        hist = [0 for _ in range(5)]

    if args.prefix == 'human':
        same_ans = 0
        all_same_eff = 0
        all_same_num = 0
        total_incomplete = 0
        inc_avg_eff = 0

    check = []

    for ids in dials:
        entry = dials[ids]
        if args.prefix == 'human':
            all_the_same = True
        removed_target = False

        entry_avg_eff = 0

        #lenght = len(entry['qas'])
        if args.lenght == -2:
            lenght = len(entry)
            stuff = entry
        else:
            lenght = len(entry[:args.lenght])
            stuff = entry[:args.lenght]

        state = states[ids]['state']
        #total_questions += len(entry[:args.lenght])
        total_questions += lenght

        total_dial += 1

        for q in stuff:
            #for q in entry['qas']:
            if q['effective']:
                total_effective += 1
                entry_avg_eff += 1
                if args.prefix != 'human':
                    hist[pos(entry.index(q))] += 1

            if args.prefix == 'human':
                if q['ans'] == q['model_ans']:
                    same_ans += 1 
                else:
                    all_the_same = False

            removed_target = removed_target or (q['removed_target'] == 'yes')
            if q['removed_target'] == 'yes':
                assert(removed_target)
            if q['removed_target'] == 'no':
                assert(not removed_target)

        #entry_avg_eff /= len(entry[:args.lenght])
        entry_avg_eff /= lenght
        per_dial_effective_avg += entry_avg_eff

        #if entry['state']:
        if state == 'success':
            total_success += 1
            succ_avg_eff += entry_avg_eff
        #else:
        elif state == 'failure':
            total_fail += 1
            fail_avg_eff += entry_avg_eff
        elif state == 'incomplete':
            total_incomplete += 1
            inc_avg_eff += entry_avg_eff

        if args.prefix == 'human' and all_the_same:
            all_same_num += 1
            all_same_eff += entry_avg_eff

        rm_target_dial += 1 if removed_target else 0
        assert(total_effective <= total_questions)

        if (not removed_target):
            check.append(int(ids))
                
    per_dial_effective_avg /= len(dials)

    np.savetxt('ooi_target-notsame.txt', np.array(check))

    print('----- Summary of Effectiveness of {} -----'.format(args.prefix))
    print('Total # questions: {}'.format(total_questions))
    print('Total # dialogues: {}'.format(total_dial))
    print('Total # successful dialogues: {}'.format(total_success))
    print('Total # failed dialogues: {}'.format(total_fail))
    if args.prefix == 'human':
        print('Total # incomplete dialogues: {}'.format(total_incomplete))
    print('Tast success: {}'.format(total_success/total_dial))
    print('Total # effective questions: {}'.format(total_effective))
    print('{0:.2f}% of effecffective questions'.format(total_effective/total_questions*100))
    print('Average effectiveness per dialogue: {0:.4f}'.format(per_dial_effective_avg))
    print("Total # dialogues where the target was removed: {}".format(rm_target_dial))
    try:
        print('Average effectiveness in sucessful dialogues: {0:.4f}'.format(succ_avg_eff/ total_success))
    except ZeroDivisionError:
        pass
    try: 
        print('Average effectiveness in failed dialogues: {0:.4f}'.format(fail_avg_eff / total_fail))
    except ZeroDivisionError:
        pass
    if args.prefix == 'human':
        try: 
            print('Average effectiveness in incomplete dialogues: {0:.4f}'.format(inc_avg_eff / total_incomplete))
        except ZeroDivisionError:
            pass

    if args.prefix == 'human':
        print('{0:.2f}% of questions got the same answer from humans and the oracle'.format(same_ans/total_questions*100)) 
        print('Total # of dialogues where both answers where the same: {}'.format(all_same_num))
        print('Average effectiveness where both answers where the same: {0:.4f}'.format(all_same_eff/all_same_num))

    #else:
    #   
        #N = 4
        #fig, ax = plt.subplots()

        #ind = np.arange(N)    # the x locations for the groups
        #width = 0.8         # the width of the bars
        #p1 = ax.bar(ind, hist, width)

        #ax.set_title('Effectiveness by position in the dialogue history in ' + args.prefix)
        #ax.set_xticks(ind)
        #ax.set_xticklabels(('1-2', '3-4', '5-6', '7-8'))
        #ax.set_ylim([0, 30000])

        #ax.autoscale_view()

        #plt.savefig(args.prefix + '_eff_hist.png')
 
    #TODO dump summary to file
    if args.to_file:
        pass
