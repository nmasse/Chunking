import numpy as np
import sys, os
import time
from analysis import *

savedir = './savedir/'
file = 'RF_cue_sequence_cue_p4_100_neuron_high_lr_v0_acc98.pkl'
base = 'RF_cue_sequence_cue_p4_100_neuron_high_lr_v0_acc'
addition = ['90.pkl', '95.pkl', '96.pkl', '97.pkl', '98.pkl']


for a in addition:

    file = base + a
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

f = 'debug_sequence_sequence_cue_RF_detection_RF_cue_p5_v0acc80_regenerated_acc80.pkl'

print('Analyzing network',f,'...')
save_fn = './savedir/new/analysis_' + f
analyzed.append(f)

try:
    analyze_model_from_file(path + f, savefile=save_fn)
except KeyboardInterrupt:
    quit('Quit by KeyboardInterrupt')

quit()
"""
"""
for i in range(1):
    for f in file_list:
        #if ('acc90_.pkl' in f or 'acc95_.pkl' in f) and (f not in analyzed):
        if 'sequence_sequence_cue_4' in f:
            print('Analyzing network',f,'...')
            save_fn = './savedir/new/analysis_' + f
            analyzed.append(f)

            try:
                analyze_model_from_file(path + f, savefile=save_fn)
            except KeyboardInterrupt:
                quit('Quit by KeyboardInterrupt')

            quit()
    #time.sleep(1800)
"""
