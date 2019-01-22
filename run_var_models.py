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

num_pulses = [5,6]
var_delay = [False] #[True]

for n in num_pulses:
    for delay in var_delay:
        print('Training network with variable delay {} with {} pulses, without cue...'.format(delay, n))
        save_fn = 'restart_var_delay_{}.pkl'.format(n) if delay else 'restart_no_var_delay_{}.pkl'.format(n)
        updates = {'num_pulses': n, 'var_delay': delay, 'var_resp_delay': delay, 'var_num_pulses': False, 'save_fn': save_fn, 'order_cue': False}
        update_parameters(updates)
        try_model(gpu_id)


