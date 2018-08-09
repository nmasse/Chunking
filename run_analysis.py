import numpy as np
from parameters import *
import model
import sys
from analysis import *

task = "chunking"

file_list = ['chunking_6_cue_off.pkl','chunking_8_cue_on.pkl']

for file in file_list:
    print('Analyzing network...')
    save_fn = 'shuffling_' + file
    analyze_model_from_file(file, savefile = save_fn, analysis = False)
