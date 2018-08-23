import numpy as np
import sys, os
from analysis import *

task = "chunking"

path = './savedir/one_hot_perfect/'
file_list = ['sequence_cue_4_all_RF.pkl']#os.listdir(path)

for f in file_list:
    print('Analyzing network...')
    save_fn = './savedir/one_hot_perfect/analysis_' + f
    analyze_model_from_file(path + f, savefile=save_fn)
