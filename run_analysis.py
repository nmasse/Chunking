import numpy as np
from parameters import *
import model
import sys
from analysis import *

task = "chunking"

file_list = ['./savedir/sequence_cue_6_var_delay_all_RF.pkl']

for file in file_list:
    print('Analyzing network...')
    save_fn = 'analysis_' + file
    analyze_model_from_file(file, savefile=save_fn)
