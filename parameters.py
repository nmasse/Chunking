import numpy as np
import tensorflow as tf
import os
import pickle

print("--> Loading parameters...")

"""
Independent parameters
"""

par = {
    # Setup parameters
    'save_dir'              : './savedir_standard/',
    'save_fn'               : 'model_results.pkl',
    'weight_load_fn'        : './savedir_standard/weights.pkl',
    'load_prev_weights'     : False,
    'analyze_model'         : True,
    'balance_EI'            : True,

    # Network configuration
    'synapse_config'        : 'std_stf', # Full is 'std_stf'
    'exc_inh_prop'          : 0.8,       # Literature 0.8, for EI off 1
    'connection_prob'       : 0.25,
    'response_multiplier'   : 4,
    'tol'                   : 0.2,
    'architecture'          : 'BIO',

    # Task parameters (non-timing)
    'trial_type'            : ['sequence'],
    'var_delay'             : True,
    'var_delay_scale'       : 12,        # Set for 9% to 15% catch trials for RF
    'var_num_pulses'        : False,
    'all_RF'                : True,
    'num_pulses'            : 4,
    'pulse_prob'            : 1,

    # Network shape
    'num_motion_tuned'      : 24,
    'num_fix_tuned'         : 0,
    'num_RFs'               : 1,
    'n_hidden'              : 100,
    'output_type'           : 'one_hot',

    # Timings and rates
    'dt'                    : 20,
    'learning_rate'         : 4e-3,
    'membrane_time_constant': 100,

    # Variance values
    'clip_max_grad_val'     : 1,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0,
    'noise_rnn_sd'          : 0.1,

    # Tuning function data
    'num_motion_dirs'       : 8,
    'tuning_height'         : 4,        # magnitutde scaling factor for von Mises
    'kappa'                 : 2,        # concentration scaling factor for von Mises

    # Cost parameters
    'spike_cost'            : 1e-9,     # run another with 1e-4
    'wiring_cost'           : 0.,

    # Synaptic plasticity specs
    'tau_fast'              : 200,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_train_size'      : 1024,
    'num_iterations'        : 10000000,
    'iters_between_outputs' : 50,

    # Task specs
    'dead_time'             : 0,    # Time at start of trial that is masked
    'fix_time'              : 200,  # Amount of fixation time before stimulus is shown
    'sample_time'           : 200,  # Sample time for sequence tasks
    'sample_time_RF'        : 500,  # Sample time for RF-based tasks
    'delay_time'            : 200,  # Short delay period (augmented for pulses)
    'long_delay_time'       : 200,  # Long delay period
    'RF_long_delay_time'    : 1000, # Long delay period for RF tasks
    'var_delay_max'         : 500,  # Maximum delay caused by var delay
    'resp_cue_time'         : 200,  # Duration of a requested response
    'mask_duration'         : 40,   # Duration of training mask after test onset
    'num_rules'             : 1,    # Legacy, used in analysis.py

    # Analysis
    'svm_normalize'         : True,
    'decoding_reps'         : 5,
    'decode_test'           : False,
    'decode_rule'           : False,
    'decode_sample_vs_test' : False,
}


"""
Dependent parameters
"""

def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """

    print('Updating parameters...')
    for key, val in updates.items():
        par[key] = val

    update_dependencies()
    if par['load_prev_weights']:
        load_previous_weights()


def load_previous_weights():

    x = pickle.load(open(par['weight_load_fn'],'rb'))
    par['h_init'] = x['weights']['h_init']
    par['W_in_init']  = x['weights']['W_in']
    par['W_rnn_init'] = x['weights']['W_rnn']
    par['W_out_init'] = x['weights']['W_out']
    par['b_rnn_init'] = x['weights']['b_rnn']
    par['b_out_init'] = x['weights']['b_out']
    print('Weights from {} loaded.'.format(par['weight_load_fn']))


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    # Backwards compatibility
    if 'pulse_prob' not in par.keys():
        par['pulse_prob'] = 1.

    if par['num_pulses'] != par['num_RFs']:
        print('WARNING:  NUMBER OF PULSES IS NOT EQUAL TO THE NUMBER OF RECEPTIVE FIELDS')

    # Set up a variety of delay times for the appropriate task types
    par['delay_times'] = par['delay_time']*np.ones((par['num_pulses']), dtype = np.int16)
    if par['var_delay']:
        par['delay_times'][1::3] += 100
        par['delay_times'][2::3] -= 100

    # Determine the number of time steps
    time_steps_sequence = par['dead_time'] + par['fix_time'] \
        + par['num_pulses']*par['sample_time'] \
        + (2*par['num_pulses']-1)*par['sample_time'] \
        + par['long_delay_time'] \
        + np.sum(par['delay_times'])

    time_steps_RFs = par['dead_time'] + par['fix_time'] \
        + 2*par['sample_time_RF'] \
        + par['RF_long_delay_time'] \
        + par['var_delay_max'] \
        + par['dt']

    par['num_time_steps'] = int(np.maximum(time_steps_sequence, time_steps_RFs)//par['dt'])

    # Number of input neurons
    par['num_max_pulse'] = par['num_pulses']
    par['num_cue_tuned'] = 1 #par['num_max_pulse']
    par['total_motion_tuned'] = par['num_motion_tuned']*par['num_RFs']

    if len(par['trial_type']) > 1:
        par['num_rule_tuned'] = len(par['trial_type'])
    else:
        par['num_rule_tuned'] = 0

    par['n_input'] = par['total_motion_tuned'] + par['num_fix_tuned'] + par['num_cue_tuned'] + par['num_rule_tuned']

    # Adjust output size and loss function based on output type
    if par['output_type'] == 'directional':
        par['n_output'] = 2
        par['loss_function'] = 'MSE'
    elif par['output_type'] == 'one_hot':
        par['n_output'] = par['num_motion_dirs'] + 1
        par['loss_function'] = 'cross_entropy'

    # General network shape
    par['shape'] = [par['n_input'], par['n_hidden'], par['n_output']]

    # If num_inh_units is set > 0, then neurons can be either excitatory or
    # inihibitory; is num_inh_units = 0, then the weights projecting from
    # a single neuron can be a mixture of excitatory or inhibitory
    if par['exc_inh_prop'] < 1:
        par['EI'] = True
    else:
        par['EI']  = False

    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']

    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    par['EI_list'][-par['num_inh_units']:] = -1.

    par['drop_mask'] = np.ones((par['n_hidden'],par['n_hidden']), dtype=np.float32)
    par['ind_inh'] = np.where(par['EI_list']==-1)[0]
    par['drop_mask'][:, par['ind_inh']] = 0.
    par['drop_mask'][par['ind_inh'], :] = 0.

    par['EI_matrix'] = np.diag(par['EI_list'])

    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']

    # The standard deviation of the Gaussian noise added to each RNN neuron at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd']

    par['dt_sec'] = par['dt']/1000

    ####################################################################
    ### Setting up assorted intial weights, biases, and other values ###
    ####################################################################

    par['h_init'] = 0.1*np.ones((par['batch_train_size'], par['n_hidden']), dtype=np.float32)

    par['input_to_hidden_dims'] = [par['n_hidden'], par['n_input']]
    par['hidden_to_hidden_dims'] = [par['n_hidden'], par['n_hidden']]


    # Initialize input weights
    par['W_in_init'] = initialize([par['n_input'], par['n_hidden']], shape=.2)

    # Initialize starting recurrent weights
    # If excitatory/inhibitory neurons desired, initializes with random matrix with
    #   zeroes on the diagonal
    # If not, initializes with a diagonal matrix
    if par['EI']:
        par['W_rnn_init'] = initialize([par['n_hidden'], par['n_hidden']])

        if par['balance_EI']:
            par['W_rnn_init'][:, par['ind_inh']] = initialize([par['n_hidden'], par['num_inh_units']], shape=.2, scale=1.)
            par['W_rnn_init'][par['ind_inh'], :] = initialize([par['n_hidden'], par['num_inh_units']], shape=.2, scale=1.)

        for i in range(par['n_hidden']):
            par['W_rnn_init'][i,i] = 0
        par['w_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'])
    else:
        par['W_rnn_init'] = 0.54*np.eye(par['n_hidden'])
        par['w_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32)

    par['b_rnn_init'] = np.zeros((1, par['n_hidden']), dtype=np.float32)

    # Effective synaptic weights are stronger when no short-term synaptic plasticity
    # is used, so the strength of the recurrent weights is reduced to compensate

    if par['synapse_config'] == None:
        par['W_rnn_init'] = par['W_rnn_init']/(spectral_radius(par['W_rnn_init']))


    # Initialize output weights and biases
    par['W_out_init'] =initialize([par['n_hidden'], par['n_output']])
    par['b_out_init'] = np.zeros((1, par['n_output']), dtype=np.float32)
    par['w_out_mask'] = np.ones((par['n_hidden'], par['n_output']), dtype=np.float32)

    # if par['EI']:
    #     par['ind_inh'] = np.where(par['EI_list'] == -1)[0]
    #     par['W_out_init'][:, par['ind_inh']] = 0
    #     par['w_out_mask'][:, par['ind_inh']] = 0

    """
    Setting up synaptic parameters
    0 = static
    1 = facilitating
    2 = depressing
    """
    par['synapse_type'] = np.zeros(par['n_hidden'], dtype=np.int8)

    # only facilitating synapses
    if par['synapse_config'] == 'stf':
        par['synapse_type'] = np.ones(par['n_hidden'], dtype=np.int8)

    # only depressing synapses
    elif par['synapse_config'] == 'std':
        par['synapse_type'] = 2*np.ones(par['n_hidden'], dtype=np.int8)

    # even numbers facilitating, odd numbers depressing
    elif par['synapse_config'] == 'std_stf':
        par['synapse_type'] = np.ones(par['n_hidden'], dtype=np.int8)
        par['ind'] = range(1,par['n_hidden'],2)
        par['synapse_type'][par['ind']] = 2

    par['alpha_stf'] = np.ones((par['n_hidden'], 1), dtype=np.float32)
    par['alpha_std'] = np.ones((par['n_hidden'], 1), dtype=np.float32)
    par['U'] = np.ones((par['n_hidden'], 1), dtype=np.float32)

    # initial synaptic values
    par['syn_x_init'] = np.zeros((par['n_hidden'], par['batch_train_size']), dtype=np.float32)
    par['syn_u_init'] = np.zeros((par['n_hidden'], par['batch_train_size']), dtype=np.float32)

    for i in range(par['n_hidden']):
        if par['synapse_type'][i] == 1:
            par['alpha_stf'][i,0] = par['dt']/par['tau_slow']
            par['alpha_std'][i,0] = par['dt']/par['tau_fast']
            par['U'][i,0] = 0.15
            par['syn_x_init'][i,:] = 1
            par['syn_u_init'][i,:] = par['U'][i,0]

        elif par['synapse_type'][i] == 2:
            par['alpha_stf'][i,0] = par['dt']/par['tau_fast']
            par['alpha_std'][i,0] = par['dt']/par['tau_slow']
            par['U'][i,0] = 0.45
            par['syn_x_init'][i,:] = 1
            par['syn_u_init'][i,:] = par['U'][i,0]

    par['syn_x_init'] = par['syn_x_init'].T
    par['syn_u_init'] = par['syn_u_init'].T
    par['alpha_stf'] = par['alpha_stf'].T
    par['alpha_std'] = par['alpha_std'].T
    par['U'] = par['U'].T


    if par['architecture'] == 'LSTM':
        c = 0.01
        par['Wf_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_input'], par['n_hidden']]))
        par['Wi_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_input'], par['n_hidden']]))
        par['Wo_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_input'], par['n_hidden']]))
        par['Wc_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_input'], par['n_hidden']]))

        par['Uf_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['Ui_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['Uo_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['Uc_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))


        par['bf_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)
        par['bi_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)
        par['bo_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)
        par['bc_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)

def initialize(dims, shape=0.1, scale=1.0):
    w = np.random.gamma(shape, scale, size=dims)
    w *= (par['connection_prob'] > np.random.rand(*dims))
    return np.float32(w)


def spectral_radius(A):
    return np.max(abs(np.linalg.eigvals(A)))


update_dependencies()
print("--> Parameters successfully loaded.\n")
