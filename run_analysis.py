import numpy as np
import sys
from analysis import *

task = "chunking"

file_list = ['./savedir/sequence_4_all_RF_low_conn.pkl']

for file in file_list:
    print('Analyzing network...')
    save_fn = 'analysis_' + file
    analyze_model_from_file(file, savefile=save_fn)
