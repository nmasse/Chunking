import difflib
import pickle
import os

def record_code_state():
    code_state = {}
    code_state['model.py'] = open('./model.py', 'r').readlines()
    code_state['parameters.py'] = open('./parameters.py', 'r').readlines()
    code_state['stimulus.py'] = open('./stimulus.py', 'r').readlines()
    code_state['AdamOpt.py'] = open('./AdamOpt.py', 'r').readlines()
    code_state['analysis.py'] = open('./analysis.py', 'r').readlines()

    return code_state


def diff_state_and_current(saved_code_state):

    new_code_state = {}
    new_code_state['model.py'] = open('./model.py', 'r').readlines()
    new_code_state['parameters.py'] = open('./parameters.py', 'r').readlines()
    new_code_state['stimulus.py'] = open('./stimulus.py', 'r').readlines()
    new_code_state['AdamOpt.py'] = open('./AdamOpt.py', 'r').readlines()
    new_code_state['analysis.py'] = open('./analysis.py', 'r').readlines()

    print('Code Diff Instructions:\n  - indicates something in the saved version.\n  + indicates something in the current version.')
    print('  No +/- lines indicates no differences.')
    #print('  If the differences seem too great to update by hand,\n  use "historian.load_code_state" to load saved version.')

    for k in saved_code_state.keys():
        result = difflib.unified_diff(saved_code_state[k], new_code_state[k])

        if len(result) > 0:
            print('\n' + '-'*60 + '\n' + k + '\n' + '-'*60)
            for l in result:
                print(l[:-1])



# CODE SAVING AND LOADING IN PROGRESS
"""
def load_code_state(code_state):

    current_code_state = {}
    current_code_state['model.py'] = open('./model.py', 'r').readlines()
    current_code_state['parameters.py'] = open('./parameters.py', 'r').readlines()
    current_code_state['stimulus.py'] = open('./stimulus.py', 'r').readlines()
    current_code_state['AdamOpt.py'] = open('./AdamOpt.py', 'r').readlines()
    current_code_state['analysis.py'] = open('./analysis.py', 'r').readlines()

    pickle.dump(current_code_state, open('code_tmp.pkl', 'rb'))

    for k in code_state.keys():
        f = open('./'+k, 'w')
        for l in code_state[k]:
            f.write(l)
"""
