import json

def get_state(pgame): 
    return pgame.lower().split()[-1][:-1]

if __name__ == "__main__":
    
    with open('data/combined_10Q.json', 'r') as fl:
        played_games = json.load(fl)

    cldial = {}
    rldial = {}
    sldial = {}
    bldial = {}

    for entry in played_games:
        cltext = entry['CLText1'][13:].strip()
        cldial[str(entry['game_id'])] = {'gen_dialogue': cltext[:cltext.index('<<<<<')],
                                         'state': get_state(entry['CLText1'])}

        rltext = entry['RLText'][14:].strip()
        rldial[str(entry['game_id'])] = {'gen_dialogue': rltext[:rltext.index('<<<<<')],
                                         'state': get_state(entry['RLText'])}

        sltext = entry['SLText'][9:].strip()
        sldial[str(entry['game_id'])] = {'gen_dialogue': sltext[:sltext.index('<<<<<')],
                                         'state': get_state(entry['SLText'])}

        bltext = entry['changeAllText'][9:].strip()
        bldial[str(entry['game_id'])] = {'gen_dialogue': bltext[:bltext.index('<<<<<')],
                                         'state': get_state(entry['changeAllText'])}


    with open('cl1_dialogues-10Q.json', 'w') as fl:
        json.dump(cldial, fl)

    with open('rl1_dialogues-10Q.json', 'w') as fl:
        json.dump(rldial, fl)

    with open('sl_dialogues-10Q.json', 'w') as fl:
        json.dump(sldial, fl)

    with open('bl_dialogues-10Q.json', 'w') as fl:
        json.dump(bldial, fl)
