import numpy as np
from parameters import *
import model
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import random as rd
import pickle
from plot import *


def try_model(gpu_id):

    try:
        # Run model
        model.main(gpu_id)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')

def plot(filename, num_pulses, savename):
    x = pickle.load(open('./savedir/'+filename, 'rb'))
    color = [(rd.uniform(0,1),rd.uniform(0,1),rd.uniform(0,1)) for i in range(num_pulses)]
    for i in range(num_pulses):
        plt.plot(np.mean(x['synaptic_sample_decoding'][0,i,:,:], axis = 0), color =color[i])
        plt.bar(np.array(x['timeline'])[np.arange(-2, -2*n-1, -2)], height = 0.04, width = 1, color = 'k')
        plt.bar(np.array(x['timeline'])[np.arange(1,2*n+1,2)], height = 0.04, width = 1, color = 'k')
        #bar(x, height, width, *, align='center', **kwargs)
    plt.savefig("./savedir/"+savename+"_synaptic.png")
    plt.close()

    for i in range(num_pulses):
        plt.plot(np.mean(x['neuronal_sample_decoding'][0,i,:,:], axis = 0), color =color[i])
        plt.bar(np.array(x['timeline'])[np.arange(-2, -2*n-1, -2)], height = 0.04, width = 1, color = 'k')
        plt.bar(np.array(x['timeline'])[np.arange(1,2*n+1,2)], height = 0.04, width = 1, color = 'k')
    plt.savefig("./savedir/"+savename+"_neuronal.png")
    plt.close()

    for i in range(num_pulses):
        plt.plot(np.mean(x['combined_decoding'][0,i,:,:], axis = 0), color =color[i])
        plt.bar(np.array(x['timeline'])[np.arange(-2, -2*n-1, -2)], height = 0.04, width = 1, color = 'k')
        plt.bar(np.array(x['timeline'])[np.arange(1,2*n+1,2)], height = 0.04, width = 1, color = 'k')
    plt.savefig("./savedir/"+savename+"_combined.png")
    plt.close()


# Second argument will select the GPU to use
# Don't enter a second argument if you want TensorFlow to select the GPU/CPU
try:
    gpu_id = sys.argv[1]
    print('Selecting GPU ', gpu_id)
except:
    gpu_id = None


trial_type = 'sequence'
num_pulses = [3,6]
load_weights = False

# Running Task1 and Task2
"""
for pulse in num_pulses:
    for v in range(2):
        print('Training network on ', trial_type, ' task, ', pulse, ' pulses, without cue, version ', v,'...')
        save_fn = str(trial_type) + '_' + str(pulse) + '_var_delay_cue_off_v' + str(v) + '.pkl'

        updates = {'save_fn': save_fn, 'num_pulses': pulse, 'num_max_pulse': pulse, 'order_cue': False, 'num_RFs': pulse, 'trial_type': trial_type,\
        'load_prev_weights': load_weights, 'var_num_pulses': False, 'var_delay': True, 'var_resp_delay':False, 'all_RF': v}
        update_parameters(updates)

        try_model(gpu_id)
        print('\n\n\n')#"""

# Running sequence_cue
"""
trial_type = 'sequence_cue'
for pulse in num_pulses:
    for v in range(1):
        print('Training network on ', trial_type, ' task, ', pulse, ' pulses, without cue, version ', v,'...')
        save_fn = str(trial_type) + '_' + str(pulse) + '_var_delay_cue_off_v' + str(v) + '.pkl'

        updates = {'save_fn': save_fn, 'num_pulses': pulse, 'num_max_pulse': pulse, 'order_cue': False, 'num_RFs': pulse, 'trial_type': trial_type,\
        'load_prev_weights': load_weights, 'var_num_pulses': False, 'var_delay': True, 'var_resp_delay':False, 'all_RF': False}
        update_parameters(updates)

        try_model(gpu_id)"""

# Running RF_detection
"""
trial_type = 'RF_detection'
for pulse in num_pulses:
    for v in range(1):

        print('Training network on ', trial_type, ' task, ', pulse, ' pulses, without cue, version ', v,'...')
        save_fn = str(trial_type) + '_' + str(pulse) + '_var_delay_cue_off_v' + str(v) + '.pkl'

        updates = {'save_fn': save_fn, 'num_pulses': pulse, 'num_max_pulse': pulse, 'order_cue': False, 'num_RFs': pulse, 'trial_type': trial_type,\
        'load_prev_weights': load_weights, 'var_num_pulses': False, 'var_delay': True, 'var_resp_delay':False, 'all_RF': False}
        update_parameters(updates)

        try_model(gpu_id)
        #"""

# Running RF_cue
#"""
trial_type = 'RF_cue'
for pulse in num_pulses:
    for v in range(1):

        print('Training network on ', trial_type, ' task, ', pulse, ' pulses, without cue, version ', v,'...')
        save_fn = str(trial_type) + '_' + str(pulse) + '_var_delay_cue_off_v' + str(v) + '.pkl'

        updates = {'save_fn': save_fn, 'num_pulses': pulse, 'num_max_pulse': pulse, 'order_cue': False, 'num_RFs': pulse, 'trial_type': trial_type,\
        'load_prev_weights': load_weights, 'var_num_pulses': False, 'var_delay': True, 'var_resp_delay':False, 'all_RF': False}
        update_parameters(updates)

        try_model(gpu_id)#"""
