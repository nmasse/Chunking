import numpy as np
from parameters import *
import pickle
import os
import stimulus
import matplotlib.pyplot as plt
import matplotlib as mpl

def run_model(x, hidden_init, syn_x_init, syn_u_init, weights, suppress_activity = None):

    """
    Run the reccurent network
    History of hidden state activity stored in self.hidden_state_hist
    """
    hidden_state_hist, syn_x_hist, syn_u_hist = \
        rnn_cell_loop(x, hidden_init, syn_x_init, syn_u_init, weights, suppress_activity)

    """
    Network output
    Only use excitatory projections from the RNN to the output layer
    """
    y_hat = [np.dot(np.maximum(0,weights['w_out']), h) + weights['b_out'] for h in hidden_state_hist]

    syn_x_hist = np.stack(syn_x_hist, axis=1)
    syn_u_hist = np.stack(syn_u_hist, axis=1)
    hidden_state_hist = np.stack(hidden_state_hist, axis=1)

    return y_hat, hidden_state_hist, syn_x_hist, syn_u_hist


def rnn_cell_loop(x_unstacked, h, syn_x, syn_u, weights, suppress_activity):

    hidden_state_hist = []
    syn_x_hist = []
    syn_u_hist = []

    """
    Loop through the neural inputs to the RNN, indexed in time
    """

    for t, rnn_input in enumerate(x_unstacked):
        #print(t)
        if suppress_activity is not None:
            #print('len sp', len(suppress_activity))
            h, syn_x, syn_u = rnn_cell(np.squeeze(rnn_input), h, syn_x, syn_u, weights, suppress_activity[t])
        else:
            h, syn_x, syn_u = rnn_cell(np.squeeze(rnn_input), h, syn_x, syn_u, weights, 1)
        hidden_state_hist.append(h)
        syn_x_hist.append(syn_x)
        syn_u_hist.append(syn_u)

    return hidden_state_hist, syn_x_hist, syn_u_hist

def rnn_cell(rnn_input, h, syn_x, syn_u, weights, suppress_activity):

    if par['EI']:
        # ensure excitatory neurons only have postive outgoing weights,
        # and inhibitory neurons have negative outgoing weights
        W_rnn_effective = np.dot(np.maximum(0,weights['w_rnn']), par['EI_matrix'])
    else:
        W_rnn_effective = weights['w_rnn']


    """
    Update the synaptic plasticity paramaters
    """
    if par['synapse_config'] == 'std_stf':
        # implement both synaptic short term facilitation and depression
        syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
        syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
        syn_x = np.minimum(1, np.maximum(0, syn_x))
        syn_u = np.minimum(1, np.maximum(0, syn_u))
        h_post = syn_u*syn_x*h

    elif par['synapse_config'] == 'std':
        # implement synaptic short term derpression, but no facilitation
        # we assume that syn_u remains constant at 1
        syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_x*h
        syn_x = np.minimum(1, np.maximum(0, syn_x))
        h_post = syn_x*h

    elif par['synapse_config'] == 'stf':
        # implement synaptic short term facilitation, but no depression
        # we assume that syn_x remains constant at 1
        syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
        syn_u = np.minimum(1, np.maximum(0, syn_u))
        h_post = syn_u*h

    else:
        # no synaptic plasticity
        h_post = h

    """
    Update the hidden state
    All needed rectification has already occured
    """

    h = np.maximum(0, h*(1-par['alpha_neuron'])
                   + par['alpha_neuron']*(np.dot(np.maximum(0,weights['w_in']), np.maximum(0, rnn_input))
                   + np.dot(W_rnn_effective, h_post) + weights['b_rnn'])
                   + np.random.normal(0, par['noise_rnn'],size=(par['n_hidden'], h.shape[1])))

    h *= suppress_activity

    return h, syn_x, syn_u

def analyze(x,filename):
    update_parameters(x['parameters'])

    v_neuronal = np.zeros([par['num_pulses'], par['num_motion_dirs'], par['n_hidden']])
    v_synaptic = np.zeros([par['num_pulses'], par['num_motion_dirs'], par['n_hidden']])
    
    stim = stimulus.Stimulus()
    trial_info = stim.generate_trial()
    input_data = np.squeeze(np.split(trial_info['neural_input'], x['parameters']['num_time_steps'], axis=1))

    y_hat, h, syn_x, syn_u = run_model(input_data, x['parameters']['h_init'], \
        x['parameters']['syn_x_init'], x['parameters']['syn_u_init'], x['weights'])

    h = np.squeeze(np.split(h, x['parameters']['num_time_steps'], axis=1))
    syn_x = np.squeeze(np.split(syn_x, x['parameters']['num_time_steps'], axis=1))
    syn_u = np.squeeze(np.split(syn_u, x['parameters']['num_time_steps'], axis=1))

    for i in range(par['num_pulses']):
        time = x['timeline'][2*i+1] + 10
        dir = trial_info['sample'][:,i]
        for d in range(par['num_motion_dirs']):
            ind = np.where(dir==d)[0]
            v_neuronal[i,d,:] = np.mean(h[time][:,ind],axis=1)
            v_synaptic[i,d,:] = np.mean(syn_x[time][:,ind] * syn_u[time][:,ind],axis=1)

        print(np.mean(v_neuronal[i],axis=0).shape)
        v_neuronal[i] = v_neuronal[i] - np.mean(v_neuronal[i],axis=0)
        v_synaptic[i] = v_synaptic[i] - np.mean(v_synaptic[i],axis=0)

    dot_neuronal = np.zeros((par['num_pulses'], par['num_pulses'], par['num_motion_dirs']))
    for j in range(par['num_pulses']-1):
        for i in range(j+1,par['num_pulses']):
            dot_neuronal[j,i,:] = np.diag(np.dot(v_neuronal[j], np.transpose(v_neuronal[i])))

    print(dot_neuronal)

    # fig, axes = plt.subplots(nrows=1, ncols=3)
    # i = 0
    # for ax in axes.flat:
    #     im = ax.imshow(dot_neuronal[i], aspect='auto')
    #     i += 1
    # cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    # plt.colorbar(im, cax=cax, **kw)
    # plt.savefig("./savedir/chunking_analysis/"+filename[:-4]+".png")

if __name__ == "__main__":
    path = "./savedir/chunking/"
    files = os.listdir(path)
    for file in files:
        print(file)
        #file = 'chunking_8_cue_on.pkl'
        data = pickle.load(open(path+file,'rb'))
        analyze(data, file)






