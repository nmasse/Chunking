import numpy as np
import sys, os
from analysis import *

task = "chunking"

path = './savedir/one_hot_perfect/'
file_list = ['sequence_sequence_cue_4_all_RF.pkl']
file_list = ['RF_cue_sequence_cue_sequence_4_all_RF_long_delacc90_ay.pkl']

for f in file_list:
    print('Analyzing network...')
    save_fn = './savedir/analysis_' + f

    try:
        analyze_model_from_file(path + f, savefile=save_fn)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')
