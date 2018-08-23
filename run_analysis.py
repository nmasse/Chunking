import numpy as np
import sys, os
from analysis import *

task = "chunking"

path = './savedir/one_hot_perfect/'
file_list = ['sequence_cue_4_all_RF.pkl']#os.listdir(path)

for f in file_list:
    print('Analyzing network...')
    f = path + f
    save_fn = f[-4] + '_analysis.pkl'
    analyze_model_from_file(f, savefile=save_fn)
