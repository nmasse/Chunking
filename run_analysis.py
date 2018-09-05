import numpy as np
import sys, os
import time
from analysis import *

savedir = './savedir/'
file = 'RF_cue_sequence_cue_p4_100_neuron_high_lr_v0_acc80.pkl'

print('Analyzing network {}...'.format(file))
save_fn = savedir + 'new/analysis_' + file

try:
    analyze_model_from_file(savedir+file, savefile=save_fn)
except KeyboardInterrupt:
    quit('Quit by KeyboardInterrupt')


"""
path = './savedir/'
file_list = os.listdir("./savedir/") #['RF_cue_sequence_cue_sequence_4_all_RF_long_delacc90_ay.pkl']
analyzed = []

for i in range(1):
    for f in file_list:
        #if ('acc90_.pkl' in f or 'acc95_.pkl' in f) and (f not in analyzed):
        if 'acc80' in f and 'p5' in f:
            print('Analyzing network',f,'...')
            save_fn = './savedir/new/analysis_' + f
            analyzed.append(f)

            try:
                analyze_model_from_file(path + f, savefile=save_fn)
            except KeyboardInterrupt:
                quit('Quit by KeyboardInterrupt')
    #time.sleep(1800)
"""
