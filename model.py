### Authors:  Nick, Greg, Catherine, Sylvia

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time

# Model modules
from parameters import *
import stimulus
import AdamOpt
import analysis

# Match GPU IDs to nvidia-smi command
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model:

    """ Biological RNN model for supervised learning """

    def __init__(self, input_data, target_data, mask, *args):

        # Print feedback on network data shape
        print('Stimulus shape:'.ljust(18), input_data.shape)
        print('Target shape:'.ljust(18), target_data.shape)
        print('Mask shape:'.ljust(18), mask.shape, '\n')

        # Load input activity, target data, training mask, etc.
        self.input_data  = tf.unstack(input_data, axis=0)
        self.target_data = tf.unstack(target_data, axis=0)
        self.mask        = tf.unstack(mask, axis=0)

        self.lesioned_neuron, self.cut_weight_i, self.cut_weight_j, self.h_init_replace, \
            self.syn_x_init_replace, self.syn_u_init_replace = args

        # Declare all Tensorflow variables
        self.declare_variables()

        print('Input weights not relu\'d.')
        print('Relu\'d input dendrites.')

        # Build the Tensorflow graph
        self.rnn_cell_loop()

        # Train the model
        self.optimize()


    def declare_variables(self):
        """ Declare/initialize all required variables (and some constants) """

        self.var_dict = {}

        with tf.variable_scope('init'):
            self.var_dict['h_init']     = tf.get_variable('hidden', initializer=par['h_init'], trainable=True)
            self.var_dict['syn_x_init'] = tf.get_variable('syn_x_init', initializer=par['syn_x_init'], trainable=False)
            self.var_dict['syn_u_init'] = tf.get_variable('syn_u_init', initializer=par['syn_u_init'], trainable=False)

        with tf.variable_scope('rnn'):
            self.var_dict['W_in']  = tf.get_variable('W_in', initializer=par['w_in0'])
            self.var_dict['W_rnn'] = tf.get_variable('W_rnn', initializer=par['w_rnn0'])
            self.var_dict['b_rnn'] = tf.get_variable('b_rnn', initializer=par['b_rnn0'])
            self.var_dict['b_rnn_dend_in']   = tf.get_variable('b_rnn_dend_in', initializer=par['b_rnn_dend_in0'])
            self.var_dict['b_rnn_dend_gate'] = tf.get_variable('b_rnn_dend_gate', initializer=par['b_rnn_dend_gate0'])

            if par['use_hebbian_trace']:
                self.var_dict['W_hebb_init']     = tf.get_variable('W_hebb_init', initializer=par['w_hebb_init0'], trainable=False)
                self.var_dict['hebb_beta']       = tf.get_variable('hebb_beta', initializer=par['hebb_beta0'], trainable=True)

        with tf.variable_scope('out'):
            self.var_dict['W_out'] = tf.get_variable('W_out', initializer=par['w_out0'])
            self.var_dict['b_out'] = tf.get_variable('b_out', initializer=par['b_out0'])

        self.W_rnn_eff = tf.tensordot(tf.constant(par['EI_matrix']), tf.nn.relu(self.var_dict['W_rnn']), [[1],[0]]) \
            if par['EI'] else self.var_dict['W_rnn']

        self.W_in_eff = self.var_dict['W_in'] #tf.nn.relu(self.var_dict['W_in'])

        self.W_exc  = tf.constant(par['excitatory_mask']) * self.W_rnn_eff[:,:par['n_dendrites'],:]
        self.W_gate = tf.constant(par['gating_mask']) * self.W_rnn_eff[:,:par['n_dendrites'],:]
        self.W_inh  = tf.constant(par['inhibitory_mask']) * self.W_rnn_eff[:,-1,:]

        if par['use_hebbian_trace']:
            self.zero_diag = tf.constant(par['rnn_zero_diag'])
            self.hebb_mask = tf.constant(par['excitatory_mask']) * self.zero_diag

        # Analysis-based variable manipulation commands
        self.lesion = self.var_dict['W_rnn'][:,:,self.lesioned_neuron].assign(tf.zeros_like(self.var_dict['W_rnn'][:,:,self.lesioned_neuron]))
        self.cutting = self.var_dict['W_rnn'][self.cut_weight_i, :, self.cut_weight_j].assign(0.)
        self.load_h_init = self.var_dict['h_init'].assign(self.h_init_replace)
        self.load_syn_x_init = self.var_dict['syn_x_init'].assign(self.syn_x_init_replace)
        self.load_syn_u_init = self.var_dict['syn_u_init'].assign(self.syn_u_init_replace)


    def rnn_cell_loop(self):
        """ Set up network state and execute loop through
            time to generate the network outputs """

        # Set up network state recording
        self.y_hat  = []
        self.hidden_hist = []
        self.syn_x_hist = []
        self.syn_u_hist = []
        self.w_hebb_hist = []

        # Load starting network state
        h = self.var_dict['h_init']
        syn_x = self.var_dict['syn_x_init']
        syn_u = self.var_dict['syn_u_init']
        W_hebb = self.var_dict['W_hebb_init'] if par['use_hebbian_trace'] else 0.

        self.dend_in_hist = []
        self.dend_gate_hist = []

        # Loop through the neural inputs, indexed in time
        for rnn_input in self.input_data:

            # Compute the state of the hidden layer
            h, syn_x, syn_u, W_hebb = self.recurrent_cell(h, syn_x, syn_u, rnn_input, W_hebb)

            # Record network state
            self.hidden_hist.append(h)
            self.syn_x_hist.append(syn_x)
            self.syn_u_hist.append(syn_u)
            self.w_hebb_hist.append(W_hebb)

            # Compute output state
            y = h @ tf.nn.relu(self.var_dict['W_out']) + self.var_dict['b_out']
            self.y_hat.append(y)


    def recurrent_cell(self, h, syn_x, syn_u, rnn_input, W_hebb):
        """ Using the standard biologically-inspired recurrent,
            cell compute the new hidden state """

        # Apply synaptic short-term facilitation and depression, if required
        if par['synapse_config'] == 'std_stf':
            syn_x = syn_x + par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
            syn_u = syn_u + par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
            syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_pre = syn_u*syn_x*h
        else:
            h_pre = h

        # Modify excitatory weights with Hebbian trace
        if par['use_hebbian_trace']:
            W_exc = self.W_exc + self.var_dict['hebb_beta']*self.hebb_mask*tf.nn.relu(W_hebb)
        else:
            W_exc = self.W_exc[tf.newaxis,...]*tf.ones([par['batch_train_size'],1,1,1])

        # Calculate dendritic input from excitatory and stimulus connections
        dendrite_in = tf.nn.relu(tf.tensordot(rnn_input, self.W_in_eff, [[1],[0]]) \
                    + tf.einsum('bi,bidj->bdj', h_pre, W_exc) \
                    + self.var_dict['b_rnn_dend_in'])

        # Calculate dendritic gating signal from inhibitory connections
        dendrite_gate = tf.nn.sigmoid(tf.tensordot(h_pre, self.W_gate, [[1],[0]]) + self.var_dict['b_rnn_dend_gate'])

        self.dend_in_hist.append(dendrite_in)
        self.dend_gate_hist.append(dendrite_gate)

        #  Calculate neural inhibitory signal
        inhibitory_signal = h_pre @ self.W_inh

        # Calculate new hidden state
        h_post = tf.nn.relu((1-par['alpha_neuron'])*h + par['alpha_neuron'] \
               * (tf.reduce_mean(dendrite_in*dendrite_gate, axis=1) + inhibitory_signal + self.var_dict['b_rnn']) \
               + tf.random_normal(h.shape, 0, par['noise_rnn'], dtype=tf.float32))

        # Calculate the next Hebbian weights
        if par['use_hebbian_trace']:
            quantity = tf.nn.relu(tf.einsum('bdi,bj->bidj',dendrite_gate*h_pre[:,tf.newaxis,:],h_post))
            W_hebb_next = (1-par['alpha_hebb'])*W_hebb + par['alpha_hebb']*quantity
        else:
            W_hebb_next = 0.

        return h_post, syn_x, syn_u, W_hebb_next


    def optimize(self):
        """ Calculate losses and apply corrections to model """

        # Set up optimizer and required constants
        epsilon = 1e-7
        opt = AdamOpt.AdamOpt(tf.trainable_variables(), learning_rate=par['learning_rate'])

        # Calculate task performance loss
        if par['loss_function'] == 'MSE':
            perf_loss = [m*tf.reduce_mean(tf.square(t - y)) for m, t, y \
                in zip(self.mask, self.target_data, self.y_hat)]

        elif par['loss_function'] == 'cross_entropy':
            perf_loss = [m*tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=t) for m, t, y \
                in zip(self.mask, self.target_data, self.y_hat)]

        self.perf_loss = tf.reduce_mean(tf.stack(perf_loss))

        # Calculate L2 loss on hidden state spiking activity
        self.spike_loss = tf.reduce_mean(tf.stack([par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) \
            for h in self.hidden_hist]))

        # Calculate L1 loss on weight strengths
        self.wiring_loss  = tf.reduce_sum(tf.nn.relu(self.var_dict['W_in'])) \
                          + tf.reduce_sum(tf.nn.relu(self.var_dict['W_rnn'])) \
                          + tf.reduce_sum(tf.nn.relu(self.var_dict['W_out']))
        self.wiring_loss *= par['wiring_cost']

        # Collect total loss
        self.loss = self.perf_loss + self.spike_loss + self.wiring_loss

        # Compute and apply network gradients
        self.train_op = opt.compute_gradients(self.loss)


def shuffle_trials(stim):
    trial_info = {'desired_output'  :  np.zeros((par['num_time_steps'], par['batch_train_size'], par['n_output']),dtype=np.float32),
                  'train_mask'      :  np.ones((par['num_time_steps'], par['batch_train_size']),dtype=np.float32),
                  'sample'          :  -np.ones((par['batch_train_size'], par['num_pulses']),dtype=np.int32),
                  'sample_RF'       : np.zeros((par['batch_train_size'], par['num_pulses']),dtype=np.int32),
                  'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['num_time_steps'], par['batch_train_size'], par['n_input'])),
                  'pulse_id'        :  -np.ones((par['num_time_steps'], par['batch_train_size']),dtype=np.int8),
                  'test'            : np.zeros((par['batch_train_size']),dtype=np.int32)}

    train_size = par['batch_train_size']
    par['batch_train_size'] //= len(par['trial_type'])
    leftover = train_size - par['batch_train_size']*len(par['trial_type'])

    all_ind = np.arange(train_size)
    for t_ind, t in enumerate(par['trial_type']):
        if t_ind == (len(par['trial_type'])-1):
            par['batch_train_size'] += leftover
        dict = stim.generate_trial(t, var_delay=par['var_delay'], \
            var_num_pulses=par['var_num_pulses'], all_RF=par['all_RF'], test_mode=False)
        ind = np.random.choice(all_ind, par['batch_train_size'], replace=False)
        all_ind = np.setdiff1d(all_ind,ind)

        trial_info['desired_output'][:,ind,:] = dict['desired_output']
        trial_info['train_mask'][:,ind] = dict['train_mask']
        trial_info['sample'][ind,:] = dict['sample']
        trial_info['sample_RF'][ind,:] = dict['sample_RF']
        trial_info['neural_input'][:,ind,:] = dict['neural_input']
        trial_info['pulse_id'][:,ind] = dict['pulse_id']
        trial_info['test'][ind] = dict['test']

    par['batch_train_size'] = train_size
    return trial_info

def main(gpu_id=None):
    """ Run supervised learning training """

    # Isolate requested GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset TensorFlow graph before running anything
    tf.reset_default_graph()

    # Define all placeholders
    x, y, m, l, ci, cj, h, sx, su = get_placeholders()

    # Set up stimulus and model performance recording
    stim = stimulus.Stimulus()
    model_performance = {'accuracy': [], 'pulse_accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], 'trial': []}

    # Start TensorFlow session
    with tf.Session() as sess:

        # Select CPU or GPU
        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, y, m, l, ci, cj, h, sx, su)

        # Initialize variables and start the timer
        sess.run(tf.global_variables_initializer())
        t_start = time.time()

        # Begin training loop
        print('\nStarting training...\n')
        acc_count = int(0)
        accuracy_threshold = np.array([0.0, 0.6, 0.7, 0.8, 0.9, 0.95])
        save_fn = par['save_dir'] + par['save_fn']
        save_fn_ind = save_fn[1:].find('.') - 1

        for i in range(par['num_iterations']):

            # Generate a batch of stimulus for training
            trial_info = shuffle_trials(stim)

            # Put together the feed dictionary
            feed_dict = {x:trial_info['neural_input'], y:trial_info['desired_output'], m:trial_info['train_mask']}

            # Run the model
            _, loss, perf_loss, spike_loss, y_hat, state_hist, syn_x_hist, syn_u_hist = \
                sess.run([model.train_op, model.loss, model.perf_loss, model.spike_loss, model.y_hat, \
                model.hidden_hist, model.syn_x_hist, model.syn_u_hist], feed_dict=feed_dict)

            # Calculate accuracy from the model's output
            if par['output_type'] == 'directional':
                accuracy, pulse_accuracy = analysis.get_coord_perf(trial_info['desired_output'], y_hat, trial_info['train_mask'], trial_info['pulse_id'])
            elif par['output_type'] == 'one_hot':
                accuracy, pulse_accuracy = analysis.get_perf(trial_info['desired_output'], y_hat, trial_info['train_mask'], trial_info['pulse_id'])

            # Record the model's performance
            model_performance = append_model_performance(model_performance, accuracy, pulse_accuracy, loss, perf_loss, spike_loss, (i+1)*par['batch_train_size'])

            # Save and show the model's performance
            if i%par['iters_between_outputs'] == 0: #in list(range(len(par['trial_type']))):
                print_results(i, par['trial_type'], perf_loss, spike_loss, state_hist, accuracy, pulse_accuracy)

                dend_in, dend_gate, hidden = sess.run([model.dend_in_hist, model.dend_gate_hist, model.hidden_hist], feed_dict=feed_dict)
                dend_in = np.stack(dend_in, axis=0)
                dend_gate = np.stack(dend_gate, axis=0)
                hidden = np.stack(hidden, axis=0)

                for b in range(3):

                    fig, ax = plt.subplots(3,3, figsize=(8,6))
                    p0 = ax[0,0].imshow(dend_in[:,b,0,:], aspect='auto')
                    p1 = ax[0,1].imshow(dend_gate[:,b,0,:], aspect='auto')
                    p0 = ax[1,0].imshow(dend_in[:,b,1,:], aspect='auto')
                    p1 = ax[1,1].imshow(dend_gate[:,b,1,:], aspect='auto')
                    p0 = ax[2,0].imshow(dend_in[:,b,2,:], aspect='auto')
                    p1 = ax[2,1].imshow(dend_gate[:,b,2,:], aspect='auto')
                    p2 = ax[0,2].imshow(hidden[:,b,:], aspect='auto')

                    ax[0,0].set_title('dend_in 0')
                    ax[0,1].set_title('dend_gate 0')
                    ax[1,0].set_title('dend_in 1')
                    ax[1,1].set_title('dend_gate 1')
                    ax[2,0].set_title('dend_in 2')
                    ax[2,1].set_title('dend_gate 2')
                    ax[0,2].set_title('hidden')

                    fig.colorbar(p0,ax=ax[0,0])
                    fig.colorbar(p1,ax=ax[0,1])
                    fig.colorbar(p0,ax=ax[1,0])
                    fig.colorbar(p1,ax=ax[1,1])
                    fig.colorbar(p0,ax=ax[2,0])
                    fig.colorbar(p1,ax=ax[2,1])
                    fig.colorbar(p2,ax=ax[0,2])

                    plt.suptitle('Iter {}, Trial {}'.format(i, b))
                    plt.savefig('./plots/dend_state_iter{}_trial{}.png'.format(i,b))
                    plt.clf()
                    plt.close()

            # if i%200 in list(range(len(par['trial_type']))):
            #     weights = sess.run(model.var_dict)
            #     results = {
            #         'model_performance': model_performance,
            #         'parameters': par,
            #         'weights': weights}
            #     pickle.dump(results, open(par['save_dir'] + par['save_fn'], 'wb') )
            #     if i>=5 and all(np.array(model_performance['accuracy'][-5:]) > accuracy_threshold[acc_count]):
            #         break

            if False and i>5 and all(np.array(model_performance['accuracy'][-5:]) > accuracy_threshold[acc_count]):

                print('SAVING')

                weights = sess.run(model.var_dict)
                results = {
                    'model_performance': model_performance,
                    'parameters': par,
                    'weights': weights}
                acc_str = str(int(accuracy_threshold[acc_count]*100))
                sf = save_fn[:-4] + '_acc' + acc_str + save_fn[-4:]
                print(sf)
                pickle.dump(results, open(sf, 'wb') )
                acc_count += 1
                if acc_count >= len(accuracy_threshold):
                    break

        # If required, save the model, analyze it, and save the results
        if par['analyze_model']:
            weights = sess.run(model.var_dict)
            syn_x_stacked = np.stack(syn_x_hist, axis=1)
            syn_u_stacked = np.stack(syn_u_hist, axis=1)
            h_stacked = np.stack(state_hist, axis=1)
            trial_time = np.arange(0,h_stacked.shape[1]*par['dt'], par['dt'])
            mean_h = np.mean(np.mean(h_stacked,axis=2),axis=1)
            results = {
                'model_performance': model_performance,
                'parameters': par,
                'weights': weights,
                'trial_time': trial_time,
                'mean_h': mean_h}
            pickle.dump(results, open(par['save_dir'] + par['save_fn'], 'wb') )


def append_model_performance(model_performance, accuracy, pulse_accuracy, loss, perf_loss, spike_loss, trial_num):

    model_performance['accuracy'].append(accuracy)
    model_performance['pulse_accuracy'].append(pulse_accuracy)
    model_performance['loss'].append(loss)
    model_performance['perf_loss'].append(perf_loss)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['trial'].append(trial_num)

    return model_performance


def print_results(iter_num, trial_type, perf_loss, spike_loss, state_hist, accuracy, pulse_accuracy):

    print('Iter. {:4d}'.format(iter_num) + ' | ' + '_'.join(trial_type).ljust(20) +
      ' | Accuracy {:6.4f}'.format(accuracy) +
      ' | Perf loss {:6.4f}'.format(perf_loss) + ' | Spike loss {:6.4f}'.format(spike_loss) +
      ' | Mean activity {:6.4f}'.format(np.mean(state_hist)))
    print('Pulse accuracy ', np.round(pulse_accuracy,4))


def get_placeholders():
    x  = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_train_size'], par['n_input']], 'stim')
    y  = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_train_size'], par['n_output']], 'out')
    m  = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_train_size']], 'mask')

    l  = tf.placeholder(tf.int32, shape=[], name='lesion')
    ci = tf.placeholder(tf.int32, shape=[], name='cut_i')
    cj = tf.placeholder(tf.int32, shape=[], name='cut_j')
    h  = tf.placeholder(tf.float32, shape=par['h_init'].shape, name='h_init')
    sx = tf.placeholder(tf.float32, shape=par['syn_x_init'].shape, name='syn_x_init')
    su = tf.placeholder(tf.float32, shape=par['syn_u_init'].shape, name='syn_u_init')

    return x, y, m, l, ci, cj, h, sx, su


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            main(gpu_id=sys.argv[1])
        else:
            main()
    except KeyboardInterrupt:
        quit('\nQuit via KeyboardInterrupt.')
