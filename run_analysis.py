import numpy as np
from parameters import *
import model
import sys
from analysis import *

task = "chunking"

#file_list = ['restart_no_var_delay_6_acc97.pkl']
base = ['restart_no_var_delay_6_', 'restart_var_delay_6_', 'restart_no_var_delay_5_', 'restart_var_delay_5_']
addition = ['acc90.pkl', 'acc97.pkl']

for a in addition:
	for b in base:
#for file in file_list:
		file = b + a
		print('Analyzing network...')
		save_fn = 'analysis_' + file
		analyze_model_from_file(file, savefile = save_fn)
