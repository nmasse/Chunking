import numpy as np
from parameters import *
import model
import sys
from analysis import *

task = "chunking"

file_list = ['var_pulses_8_cue_on.pkl','var_pulses_8_cue_off.pkl']

for file in file_list:
    print('Analyzing network...')
    save_fn = 'cutting_shuffling_' + file
    analyze_model_from_file(file, savefile = save_fn, test_mode_pulse = True)
