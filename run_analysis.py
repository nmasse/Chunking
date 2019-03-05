import numpy as np
from parameters import *
import model
import sys
from analysis import *

task = "chunking"

#file_list = ['restart_no_var_delay_6_acc97.pkl']
base = ['restart_no_var_delay_6_', 'restart_var_delay_6_']
addition = ['acc97.pkl']

for a in addition:
    for b in base:
#for file in file_list:
        # file = b + a

        # file = 'var_delay_6_tc50_acc80.pkl'#nick_restart_var_delay_6_spike_cost_acc60.pkl
        file = 'tc40_5pulses_acc90.pkl'


        print('Analyzing network...')
        save_fn = 'analysis_' + file

        if pickle.load(open(par['save_dir']+file,'rb'))['parameters']['var_delay']:
            test_delay = True
        else:
            test_delay = False

        analyze_model_from_file(file, savefile = save_fn, test_mode_delay=test_delay)
        quit()