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
import historian

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

        # Build the Tensorflow graph
        self.rnn_cell_loop()

        # Train the model
        self.optimize()


    def declare_variables(self):
        """ Declare/initialize all required variables (and some constants) """

        lstm_var_prefixes   = ['Wf', 'Wi', 'Wo', 'Wc', 'Uf', 'Ui', 'Uo', 'Uc', 'bf', 'bi', 'bo', 'bc']
        bio_var_prefixes    = ['W_in', 'b_rnn', 'W_rnn']
        base_var_prefies    = ['W_out', 'b_out']
        
        self.var_dict = {}
        with tf.variable_scope('init'):
            self.var_dict['h_init']     = tf.get_variable('hidden', initializer=par['h_init'], trainable=True)
            self.var_dict['syn_x_init'] = tf.get_variable('syn_x_init', initializer=par['syn_x_init'], trainable=False)
            self.var_dict['syn_u_init'] = tf.get_variable('syn_u_init', initializer=par['syn_u_init'], trainable=False)
        
        # Add relevant prefixes to variable declaration
        prefix_list = base_var_prefies
        if par['architecture'] == 'LSTM':
            prefix_list += lstm_var_prefixes

        elif par['architecture'] == 'BIO':
            prefix_list += bio_var_prefixes

            # Analysis-based variable manipulation commands
            # self.lesion = self.var_dict['W_rnn'][:,self.lesioned_neuron].assign(tf.zeros_like(self.var_dict['W_rnn'][:,self.lesioned_neuron]))
            # self.cutting = self.var_dict['W_rnn'][self.cut_weight_i, self.cut_weight_j].assign(0.)
            # self.load_h_init = self.var_dict['h_init'].assign(self.h_init_replace)
            # self.load_syn_x_init = self.var_dict['syn_x_init'].assign(self.syn_x_init_replace)
            # self.load_syn_u_init = self.var_dict['syn_u_init'].assign(self.syn_u_init_replace)

        # Use prefix list to declare required variables and place them in a dict
        with tf.variable_scope('network'):
            for p in prefix_list:
                self.var_dict[p] = tf.get_variable(p, initializer=par[p+'_init'])

        if par['architecture'] == 'BIO':
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
        syn_x = self.var_dict['syn_x_init']
        syn_u = self.var_dict['syn_u_init']

        # Initialize network state
        if par['architecture'] == 'BIO':
            c = tf.constant(par['h_init'])
        elif par['architecture'] == 'LSTM':
            h = 0.*h
            c = 0.*h


        # Loop through the neural inputs, indexed in time
        for rnn_input in self.input_data:

            # Compute the state of the hidden layer
            h, c, syn_x, syn_u = self.recurrent_cell(h, c, syn_x, syn_u, rnn_input)

            # Record network state
            self.hidden_hist.append(h)
            self.syn_x_hist.append(syn_x)
            self.syn_u_hist.append(syn_u)

            # Compute output state
            y = h @ self.var_dict['W_out'] + self.var_dict['b_out']
            self.y_hat.append(y)


    def recurrent_cell(self, h, c, syn_x, syn_u, rnn_input):
        """ Using the appropriate recurrent cell
            architecture, compute the hidden state """

        if par['architecture'] == 'BIO':
  
            # Apply synaptic short-term facilitation and depression, if required
            if par['synapse_config'] == 'std_stf':
                syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
                syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
                syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
                syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
                h_post = syn_u*syn_x*h
            else:
                h_post = h

            # Compute hidden state
            h = tf.nn.relu((1-par['alpha_neuron'])*h \
              + par['alpha_neuron']*(rnn_input @ tf.nn.relu(self.var_dict['W_in']) + h_post @ self.W_rnn_eff + self.var_dict['b_rnn']) \
              + tf.random_normal(h.shape, 0, par['noise_rnn'], dtype=tf.float32))
            c = tf.constant(-1.)

        elif par['architecture'] == 'LSTM':

            # Compute LSTM state
            # f : forgetting gate, i : input gate,
            # c : cell state, o : output gate
            f   = tf.sigmoid(rnn_input @ self.var_dict['Wf'] + h @ self.var_dict['Uf'] + self.var_dict['bf'])
            i   = tf.sigmoid(rnn_input @ self.var_dict['Wi'] + h @ self.var_dict['Ui'] + self.var_dict['bi'])
            cn  = tf.tanh(rnn_input @ self.var_dict['Wc'] + h @ self.var_dict['Uc'] + self.var_dict['bc'])
            c   = f * c + i * cn
            o   = tf.sigmoid(rnn_input @ self.var_dict['Wo'] + h @ self.var_dict['Uo'] + self.var_dict['bo'])

            # Compute hidden state
            h = o * tf.tanh(c)
            syn_x = tf.constant(-1.)
            syn_u = tf.constant(-1.)

        return h, c, syn_x, syn_u


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
        if par['architecture'] == 'BIO':
            self.wiring_loss  = tf.reduce_sum(tf.nn.relu(self.var_dict['W_in'])) \
                              + tf.reduce_sum(tf.nn.relu(self.var_dict['W_rnn'])) \
                              + tf.reduce_sum(tf.nn.relu(self.var_dict['W_out']))
            self.wiring_loss *= par['wiring_cost']
        elif par['architecture'] == 'LSTM':
            self.wiring_loss = 0

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

def main(gpu_id=None, code_state=historian.record_code_state()):
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
        accuracy_threshold = \
            np.array([0.0, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, \
                      0.97, 0.98])
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

            # if i%200 in list(range(len(par['trial_type']))):
            #     weights = sess.run(model.var_dict)
            #     results = {
            #         'model_performance': model_performance,
            #         'parameters': par,
            #         'weights': weights}
            #     pickle.dump(results, open(par['save_dir'] + par['save_fn'], 'wb') )
            #     if i>=5 and all(np.array(model_performance['accuracy'][-5:]) > accuracy_threshold[acc_count]):
            #         break

            if i>5 and all(np.array(model_performance['accuracy'][-5:]) > accuracy_threshold[acc_count]):

                print('SAVING')

                weights = sess.run(model.var_dict)
                results = {
                    'model_performance': model_performance,
                    'parameters': par,
                    'weights': weights,
                    'code_state': code_state}
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
