import numpy as np
import sys
from analysis import *

task = "chunking"

file_list = ['./savedir/sequence_2_all_RF.pkl','./savedir/sequence_2_one_RF.pkl','./savedir/sequence_cue_2_all_RF.pkl']

for file in file_list:
    print('Analyzing network...')
    save_fn = file[-4] + '_test.pkl'
    analyze_model_from_file(file, savefile=save_fn)
