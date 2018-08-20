### Authors:  Nick, Greg, Catherine, Sylvia

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

    def __init__(self, input_data, target_data, mask):

        # Print feedback on network data shape
        print('Stimulus shape:'.ljust(18), input_data.shape)
        print('Target shape:'.ljust(18), target_data.shape)
        print('Mask shape:'.ljust(18), mask.shape)

        # Load input activity, target data, training mask, etc.
        self.input_data  = tf.unstack(input_data, axis=0)
        self.target_data = tf.unstack(target_data, axis=0)
        self.mask        = tf.unstack(mask, axis=0)

        # Declare all Tensorflow variables
        self.declare_variables()

        # Build the Tensorflow graph
        self.rnn_cell_loop()

        # Train the model
        self.optimize()


    def declare_variables(self):
        """ Declare/initialize all required variables (and some constants) """

        self.var_dict = {}

        with tf.variable_scope('init'):
            self.var_dict['h_init'] = tf.get_variable('hidden', initializer=par['h_init'], trainable=True)

        with tf.variable_scope('rnn'):
            self.var_dict['W_in']  = tf.get_variable('W_in', initializer=par['w_in0'])
            self.var_dict['W_rnn'] = tf.get_variable('W_rnn', initializer=par['w_rnn0'])
            self.var_dict['b_rnn'] = tf.get_variable('b_rnn', initializer=par['b_rnn0'])

        with tf.variable_scope('out'):
            self.var_dict['W_out'] = tf.get_variable('W_out', initializer=par['w_out0'])
            self.var_dict['b_out'] = tf.get_variable('b_out', initializer=par['b_out0'])

        self.W_rnn_eff = (tf.constant(par['EI_matrix']) @ tf.nn.relu(self.var_dict['W_rnn'])) \
            if par['EI'] else self.var_dict['W_rnn']


    def rnn_cell_loop(self):
        """ Set up network state and execute loop through
            time to generate the network outputs """

        # Set up network state recording
        self.y_hat  = []
        self.hidden_hist = []
        self.syn_x_hist = []
        self.syn_u_hist = []

        # Load starting network state
        h = self.var_dict['h_init']
        syn_x = tf.constant(par['syn_x_init'])
        syn_u = tf.constant(par['syn_u_init'])

        # Loop through the neural inputs, indexed in time
        for rnn_input in self.input_data:

            # Compute the state of the hidden layer
            h, syn_x, syn_u = self.recurrent_cell(h, syn_x, syn_u, rnn_input)

            # Record network state
            self.hidden_hist.append(h)
            self.syn_x_hist.append(syn_x)
            self.syn_u_hist.append(syn_u)

            # Compute output state
            y = h @ tf.nn.relu(self.var_dict['W_out']) + self.var_dict['b_out']
            self.y_hat.append(y)


    def recurrent_cell(self, h, syn_x, syn_u, rnn_input):
        """ Using the standard biologically-inspired recurrent,
            cell compute the new hidden state """

        # Apply synaptic short-term facilitation and depression, if required
        if par['synapse_config'] == 'std_stf':
            syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
            syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
            syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_post = syn_u*syn_x*h
        else:
            h_post = h

        # Calculate new hidden state
        h = tf.nn.relu((1-par['alpha_neuron'])*h + par['alpha_neuron'] \
            * (rnn_input @ self.var_dict['W_in'] + h_post @ self.W_rnn_eff + self.var_dict['b_rnn']) \
            + tf.random_normal(h.shape, 0, par['noise_rnn'], dtype=tf.float32))

        return h, syn_x, syn_u


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


def main(gpu_id=None):
    """ Run supervised learning training """

    # Isolate requested GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset TensorFlow graph before running anything
    tf.reset_default_graph()

    # Define all placeholders
    x = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_train_size'], par['n_input']], 'stim')
    y = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_train_size'], par['n_output']], 'out')
    m = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_train_size']], 'mask')

    # Set up stimulus and model performance recording
    stim = stimulus.Stimulus()
    model_performance = {'accuracy': [], 'pulse_accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], 'trial': []}

    # Start TensorFlow session
    with tf.Session() as sess:

        # Select CPU or GPU
        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, y, m)

        # Initialize variables and start the timer
        sess.run(tf.global_variables_initializer())
        t_start = time.time()

        # Begin training loop
        print('\nStarting training...\n')
        for i in range(par['num_iterations']):

            # Generate a batch of stimulus for training
            trial_info = stim.generate_trial(par['trial_type'], var_delay=par['var_delay'], var_resp_delay=par['var_resp_delay'], \
                var_num_pulses=par['var_num_pulses'], all_RF=par['all_RF'], test_mode=False)

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
            if i%par['iters_between_outputs'] == 0:
                print_results(i, perf_loss, spike_loss, state_hist, accuracy, pulse_accuracy)

            if i%200 == 0:
                weights = sess.run(model.var_dict)
                results = {
                    'model_performance': model_performance,
                    'parameters': par,
                    'weights': weights}
                pickle.dump(results, open(par['save_dir'] + par['save_fn'], 'wb') )
                if accuracy > 0.9:
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

            #x = pickle.load(open(par['save_dir'] + par['save_fn'], 'rb'))

            #analysis.analyze_model(x, trial_info, y_hat, state_hist, syn_x_hist, syn_u_hist, model_performance, weights, analysis = False, test_mode_pulse=False, pulse=0, test_mode_delay=False, stim_num=0,\
                #simulation = True, cut = False, lesion = False, tuning = True, decoding = True, load_previous_file = False, save_raw_data = False)


            # Generate another batch of trials with test_mode = True (sample and test stimuli
            # are independently drawn), and then perform tuning and decoding analysis
            # trial_info = stim.generate_trial(test_mode = True)
            # y_hat, state_hist, syn_x_hist, syn_u_hist = \
            #     sess.run([model.y_hat, model.hidden_hist, model.syn_x_hist, model.syn_u_hist], \
            #     {x: trial_info['neural_input'], y: trial_info['desired_output'], mask: trial_info['train_mask']})
            # analysis.analyze_model(trial_info, y_hat, state_hist, syn_x_hist, syn_u_hist, model_performance, weights, \
            #     simulation = False, lesion = False, tuning = par['analyze_tuning'], decoding = True, load_previous_file = True, save_raw_data = False)


def append_model_performance(model_performance, accuracy, pulse_accuracy, loss, perf_loss, spike_loss, trial_num):

    model_performance['accuracy'].append(accuracy)
    model_performance['pulse_accuracy'].append(pulse_accuracy)
    model_performance['loss'].append(loss)
    model_performance['perf_loss'].append(perf_loss)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['trial'].append(trial_num)

    return model_performance


def print_results(iter_num, perf_loss, spike_loss, state_hist, accuracy, pulse_accuracy):

    print('Iter. {:4d}'.format(iter_num) + ' | Accuracy {:0.4f}'.format(accuracy) +
      ' | Perf loss {:0.4f}'.format(perf_loss) + ' | Spike loss {:0.4f}'.format(spike_loss) +
      ' | Mean activity {:0.4f}'.format(np.mean(state_hist)))
    print('Pulse accuracy ', np.round(pulse_accuracy,4))


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            main(gpu_id=sys.argv[1])
        else:
            main()
    except KeyboardInterrupt:
        quit('\nQuit via KeyboardInterrupt.')
