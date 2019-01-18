import numpy as np
from parameters import *
import model
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import random as rd
import pickle

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

# num_pulses = [10, 12, 14]
# num_max_pulse = [10, 12, 14]
'''
weekends = [8]

for n in weekends:
    if par['var_delay'] and par['var_resp_delay']:
        #for n in num_pulses:
        for i in range(2):
            if i == 0:
                print('Training network with variable delay on', n, ' pulses, with cue...')
                save_fn = 'var_delay_' + str(n) + '_cue_on.pkl'
                updates = {'num_pulses': n, 'save_fn': save_fn, 'order_cue': True}
                update_parameters(updates)
                try_model(gpu_id)
                #plot(save_fn, par['num_pulses'], savename='var_delay_'+str(par['num_pulses'])+'_pulses_cue_on')
            elif i == 1:
                print('Training network with variable delay on', n, ' pulses, without cue...')
                save_fn = 'var_delay_' + str(n) + '_cue_off.pkl'
                updates = {'num_pulses': n, 'save_fn': save_fn, 'order_cue': False}
                update_parameters(updates)
                try_model(gpu_id)
                #plot(save_fn, par['num_pulses'], savename='var_delay_'+str(par['num_pulses'])+'_pulses_cue_off')
    updates = {'var_delay': False, 'var_resp_delay': False, 'var_num_pulses': True}
    update_parameters(updates)

    if par['var_num_pulses']:
        #for n in num_max_pulse:
        # for i in range(2):
        #     if i == 0:
        #         print('Training network with variable pulses on', n, ' max pulses, with cue...')
        #         save_fn = 'var_pulses_' + str(n) + '_cue_on.pkl'
        #         updates = {'num_max_pulse': n, 'save_fn': save_fn, 'order_cue': True}
        #         update_parameters(updates)
        #         try_model(gpu_id)
        #         #plot(save_fn, par['num_pulses'], savename='var_delay_'+str(par['num_pulses'])+'_pulses_cue_on')
        #     elif i == 1:
        print('Training network with variable pulses on', n, ' max pulses, without cue...')
        save_fn = 'var_pulses_' + str(n) + '_cue_off.pkl'
        updates = {'num_max_pulse': n, 'save_fn': save_fn, 'order_cue': False}
        update_parameters(updates)
        try_model(gpu_id)
                #plot(save_fn, par['num_pulses'], savename='var_delay_'+str(par['num_pulses'])+'_pulses_cue_off')
    updates = {'var_delay': True, 'var_resp_delay': True, 'var_num_pulses': False}
    update_parameters(updates)
'''


num_pulses = [4]

for n in num_pulses:

    print('Training network with variable delay on', n, ' pulses, v4...')
    save_fn = 'old_var_delay_' + str(n) + '_v4.pkl'
    updates = {'var_delay': True, 'var_num_pulses': True, 'num_pulses': n, 'save_fn': save_fn, \
                    'num_RFs': 1, 'long_delay_time': 200, 'dead_time': 0, 'num_fix_tuned': 1}
    update_parameters(updates)
    try_model(gpu_id)

'''
    print('Training network with variable delay on', n, ' pulses, v1...')
    save_fn = 'old_var_delay_' + str(n) + '_v1.pkl'
    updates = {'var_delay': True, 'var_num_pulses': True, 'num_pulses': n, 'save_fn': save_fn, \
                    'num_RFs': 1, 'long_delay_time': 200}
    update_parameters(updates)
    try_model(gpu_id)

    print('Training network with variable delay on', n, ' pulses, v2...')
    save_fn = 'old_var_delay_' + str(n) + '_v2.pkl'
    updates = {'dead_time': 0, 'save_fn': save_fn}
    update_parameters(updates)
    try_model(gpu_id)

    print('Training network with variable delay on', n, ' pulses, v3...')
    save_fn = 'old_var_delay_' + str(n) + '_v3.pkl'
    updates = {'num_fix_tuned': 1, 'save_fn': save_fn}
    update_parameters(updates)
    try_model(gpu_id)
'''
