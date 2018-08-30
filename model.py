### Authors:  Nick, Greg, Catherine, Sylvia

# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time
import matplotlib.pyplot as plt
import matplotlib.colors
plt.switch_backend('agg')

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
            self.syn_x_init_replace, self.syn_u_init_replace, self.moving_average = args

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
            self.var_dict['h_init']     = tf.get_variable('hidden', initializer=par['h_init'], trainable=True)
            self.var_dict['m_init']     = tf.get_variable('memory', initializer=par['h_init']/100, trainable=True)
            self.var_dict['syn_x_init'] = tf.get_variable('syn_x_init', initializer=par['syn_x_init'], trainable=False)
            self.var_dict['syn_u_init'] = tf.get_variable('syn_u_init', initializer=par['syn_u_init'], trainable=False)

        with tf.variable_scope('rnn'):
            self.var_dict['W_in']  = tf.get_variable('W_in', initializer=par['w_in0'])
            self.var_dict['W_rnn'] = tf.get_variable('W_rnn', initializer=par['w_rnn0'])
            self.var_dict['b_rnn'] = tf.get_variable('b_rnn', initializer=par['b_rnn0'])

        with tf.variable_scope('LTM'):
            self.var_dict['W_to_LTM']  = tf.get_variable('W_to', initializer=par['w_to_LTM0']/100)
            self.var_dict['W_fr_LTM']  = tf.get_variable('W_fr', initializer=par['w_fr_LTM0']/100)
            self.var_dict['W_rnn_LTM'] = tf.get_variable('W_rnn', initializer=par['w_rnn_LTM0']/100, trainable=False)
            self.var_dict['b_rnn_LTM'] = tf.get_variable('b_rnn', initializer=par['b_rnn_LTM0'])
            self.var_dict['W_dyn_init'] = tf.get_variable('W_dyn_init', initializer=par['w_dyn_init0'], trainable=False)

        with tf.variable_scope('out'):
            self.var_dict['W_out'] = tf.get_variable('W_out', initializer=par['w_out0'])
            self.var_dict['b_out'] = tf.get_variable('b_out', initializer=par['b_out0'])

        self.EI = tf.constant(par['EI_matrix'])

        self.W_rnn_eff = (tf.constant(par['EI_matrix']) @ tf.nn.relu(self.var_dict['W_rnn'])) \
            if par['EI'] else self.var_dict['W_rnn']

        # Analysis-based variable manipulation commands
        self.lesion = self.var_dict['W_rnn'][:,self.lesioned_neuron].assign(tf.zeros_like(self.var_dict['W_rnn'][:,self.lesioned_neuron]))
        self.cutting = self.var_dict['W_rnn'][self.cut_weight_i, self.cut_weight_j].assign(0.)
        self.load_h_init = self.var_dict['h_init'].assign(self.h_init_replace)
        self.load_syn_x_init = self.var_dict['syn_x_init'].assign(self.syn_x_init_replace)
        self.load_syn_u_init = self.var_dict['syn_u_init'].assign(self.syn_u_init_replace)

        #self.alpha = tf.get_variable('LTM_alpha', initializer=1e-9)



    def rnn_cell_loop(self):
        """ Set up network state and execute loop through
            time to generate the network outputs """

        # Set up network state recording
        self.y_hat  = []
        self.hidden_hist = []
        self.syn_x_hist = []
        self.syn_u_hist = []
        self.ltm_dyn_hist = []
        self.LTM_hist = []

        # Load starting network state
        h = self.var_dict['h_init']
        m = self.var_dict['m_init']
        syn_x = self.var_dict['syn_x_init']
        syn_u = self.var_dict['syn_u_init']
        ltm_dyn = self.var_dict['W_dyn_init']

        # Loop through the neural inputs, indexed in time
        for rnn_input in self.input_data:

            # Compute the state of the hidden layer
            if len(self.LTM_hist) < 2:
                prev = tf.zeros_like(m)
            else:
                prev = self.LTM_hist[-2]
            m_next, ltm_dyn = self.ltm_cell(m, h, prev, ltm_dyn)
            h, syn_x, syn_u = self.recurrent_cell(h, m, syn_x, syn_u, rnn_input)
            m = m_next

            # Record network state
            self.hidden_hist.append(h)
            self.syn_x_hist.append(syn_x)
            self.syn_u_hist.append(syn_u)
            self.ltm_dyn_hist.append(ltm_dyn)
            self.LTM_hist.append(m)

            # Compute output state
            y = h @ tf.nn.relu(self.var_dict['W_out']) + self.var_dict['b_out']
            self.y_hat.append(y)


    def ltm_cell(self, m, h, m_prev, ltm_dyn):

        #W_ltm = self.var_dict['W_rnn_LTM'][tf.newaxis,...] + ltm_dyn
        W_ltm = self.var_dict['W_rnn_LTM'][tf.newaxis,...]*(1 + tf.maximum(-0.5, tf.minimum(0.5,ltm_dyn)))
        W_LTM_eff = tf.einsum('ij,bjk->bik', self.EI, tf.nn.relu(W_ltm))
        W_LTM_eff = W_LTM_eff * tf.constant(par['RNN_self_conn_block'][np.newaxis,...])

        m0 = tf.einsum('bi,bij->bj', m, W_LTM_eff)
        m1 = h @ self.var_dict['W_to_LTM']
        m2 = self.var_dict['b_rnn_LTM']
        m_next = tf.nn.relu((1-par['ltm_neuron'])*m + par['ltm_neuron']*(m0 + m1 + m2) \
            + tf.random_normal(m.shape, 0, par['noise_rnn']/2, dtype=tf.float32))

        #a = par['alpha_ltm']/10
        #quantity = m[:,:,tf.newaxis] - tf.einsum('bj,bij->bij', m_next, ltm_dyn)
        #ltm_dyn_next = ltm_dyn + a * tf.einsum('bj,bij->bij', m_next, quantity)


        #quantity = tf.einsum('bi,bj->bij', (par['EI_list'][np.newaxis,:] * (m-self.moving_average))**3, m_next-self.moving_average)
        #quantity = tf.cast(self.var_dict['W_rnn_LTM'][tf.newaxis,...] > 0 , tf.float32)*quantity
        #ltm_dyn_next = (1-par['alpha_ltm'])*ltm_dyn + par['alpha_ltm']*quantity

        #pre = par['EI_list'][np.newaxis,:] * (m - self.moving_average)
        #pre = pre * tf.nn.softmax(10*pre, axis=1)
        #post = m_next - self.moving_average

        pre = par['EI_list'][np.newaxis,:] * m
        post = m_next - m

        quantity = tf.einsum('bi,bj->bij', pre, post)
        ltm_dyn_next = (1-par['alpha_ltm'])*ltm_dyn + par['alpha_ltm']*quantity #*tf.nn.softmax(10*quantity, axis=1)

        self.ltm_dyn_post = ltm_dyn_next
        self.m = m
        self.W_ltm = W_ltm
        self.W_ltm_eff = W_LTM_eff

        return m_next, ltm_dyn_next


    def recurrent_cell(self, h, m, syn_x, syn_u, rnn_input):
        """ Using the standard biologically-inspired recurrent,
            cell compute the new hidden state """

        # Apply synaptic short-term facilitation and depression, if required
        if par['synapse_config'] == 'std_stf':
            syn_x = syn_x + par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
            syn_u = syn_u + par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
            syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_post = syn_u*syn_x*h
        else:
            h_post = h

        # Calculate new hidden state
        h_next = tf.nn.relu((1-par['alpha_neuron'])*h + par['alpha_neuron'] \
            * (rnn_input @ self.var_dict['W_in'] + h_post @ self.W_rnn_eff \
            + m @ self.var_dict['W_fr_LTM'] + self.var_dict['b_rnn']) \
            + tf.random_normal(h.shape, 0, par['noise_rnn'], dtype=tf.float32))

        return h_next, syn_x, syn_u


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

        self.LTM_activity_loss = tf.reduce_mean(tf.stack([par['LTM_activity_cost']*tf.reduce_mean(tf.square(m), axis=0) \
            for m in self.LTM_hist]))

        # Calculate L1 loss on weight strengths
        self.wiring_loss  = tf.reduce_sum(tf.nn.relu(self.var_dict['W_in'])) \
                          + tf.reduce_sum(tf.nn.relu(self.var_dict['W_rnn'])) \
                          + tf.reduce_sum(tf.nn.relu(self.var_dict['W_out']))
        self.wiring_loss *= par['wiring_cost']

        # Collect total loss
        self.loss = self.perf_loss + self.spike_loss + self.wiring_loss + self.LTM_activity_loss

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
    x, y, m, l, ci, cj, h, sx, su, mav = get_placeholders()

    # Set up stimulus and model performance recording
    stim = stimulus.Stimulus()
    model_performance = {'accuracy': [], 'pulse_accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], 'trial': []}

    # Start TensorFlow session
    with tf.Session() as sess:

        # Select CPU or GPU
        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, y, m, l, ci, cj, h, sx, su, mav)

        # Initialize variables and start the timer
        sess.run(tf.global_variables_initializer())
        t_start = time.time()

        # Begin training loop
        print('\nStarting training...\n')
        acc_count = int(0)
        accuracy_threshold = np.array([0.6, 0.7, 0.8, 0.9, 0.95])
        save_fn = par['save_dir'] + par['save_fn']
        save_fn_ind = save_fn[1:].find('.') - 1

        LTM_avg = 0.1*np.ones([1,par['n_hidden']])

        for i in range(par['num_iterations']):

            # Generate a batch of stimulus for training
            trial_info = shuffle_trials(stim)

            # Put together the feed dictionary
            feed_dict = {x:trial_info['neural_input'], y:trial_info['desired_output'], m:trial_info['train_mask'], mav:LTM_avg}

            # Run the model
            _, loss, perf_loss, spike_loss, y_hat, state_hist, syn_x_hist, syn_u_hist, LTM_hist, LTM_loss = \
                sess.run([model.train_op, model.loss, model.perf_loss, model.spike_loss, model.y_hat, \
                model.hidden_hist, model.syn_x_hist, model.syn_u_hist, model.LTM_hist, model.LTM_activity_loss], feed_dict=feed_dict)

            LTM_avg = np.mean(np.mean(np.stack(LTM_hist, axis=0), axis=0), axis=0)[np.newaxis,:]

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
                print('LTM Loss:', LTM_loss)

                ltm, rnn, ltm_dyn_hist, ltm_dyn_post, mem, mem_hist, var_dict = \
                    sess.run([model.W_ltm_eff, model.W_rnn_eff, model.ltm_dyn_hist, \
                    model.ltm_dyn_post, model.m, model.LTM_hist, model.var_dict], feed_dict=feed_dict)

                print('LTM Diagnostics:')
                custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['white', 'fuchsia', 'lime', 'white'])

                #"""
                if True or i > 1000:
                    print('Plotting...')

                    for t, l in enumerate(ltm_dyn_hist):
                        break
                        plt.imshow(l[0])
                        plt.title('LTM Dynamic Weights State, Time Step {}'.format(t))
                        plt.colorbar()
                        plt.savefig('./plots/iter{}_dyn_ltm_t{}.png'.format(i, t))
                        plt.clf()
                        plt.close()

                    fig, ax = plt.subplots(1,2, figsize=[16,8])
                    im1 = ax[0].imshow(ltm[0,:,:], aspect='auto')
                    ax[0].set_title('LTM')
                    im2 = ax[1].imshow(rnn, aspect='auto')
                    ax[1].set_title('RNN')
                    fig.colorbar(im1, ax=ax[0])
                    fig.colorbar(im2, ax=ax[1])
                    plt.savefig('./plots/iter{}_ltm_rnn.png'.format(i))
                    plt.clf()
                    plt.close()
                #"""

                print('  W_to_LTM:  {:7.5f} +/- {:7.5f}'.format(np.mean(var_dict['W_to_LTM']), np.std(var_dict['W_to_LTM'])))
                print('  W_fr_LTM:  {:7.5f} +/- {:7.5f}'.format(np.mean(var_dict['W_fr_LTM']), np.std(var_dict['W_fr_LTM'])))
                print('  W_ltm_eff: {:7.5f} +/- {:7.5f}'.format(np.mean(ltm), np.std(ltm)))
                print('  W_ltm_dyn: {:7.5f} +/- {:7.5f}'.format(np.mean(ltm_dyn_hist[-1]), np.std(ltm_dyn_hist[-1])))
                print('  LTM State: {:7.5f} +/- {:7.5f}'.format(np.mean(mem), np.std(mem)))

                fig, ax = plt.subplots(1, figsize=[14,10])
                im = ax.imshow(np.mean(np.stack(mem_hist, axis=0), axis=1).T,aspect='auto')
                fig.colorbar(im, ax=ax)
                ax.set_title('Memory State Across Time')
                ax.set_xlabel('Time')
                ax.set_ylabel('Neuron')
                plt.savefig('./plots/iter{}_mem.png'.format(i))
                plt.clf()
                plt.close()

                print('Calculating PEV...')
                neuronal_pev = np.zeros([par['n_hidden'], par['num_pulses'], par['num_time_steps']], dtype=np.float32)
                neuronal_dir = np.zeros([par['n_hidden'], par['num_pulses'], par['num_time_steps']], dtype=np.float32)
                syn_efficacy = np.stack(syn_x_hist, axis=0) * np.stack(syn_u_hist, axis=0)
                sample_dir = np.ones([par['batch_train_size'], 3, par['num_pulses']])
                epsilon = 1e-9
                for p in range(par['num_pulses']):
                    sample_dir[:,1,p] = np.cos(2*np.pi*trial_info['sample'][:,p]/par['num_motion_dirs'])
                    sample_dir[:,2,p] = np.sin(2*np.pi*trial_info['sample'][:,p]/par['num_motion_dirs'])

                    for n in range(par['n_hidden']):
                        for t, mem_t in enumerate(mem_hist):

                            w            = np.linalg.lstsq(sample_dir[:,:,p], mem_t[:,n])[0][...,np.newaxis]
                            m_hat        = np.dot(sample_dir[:,:,p], w).T
                            pred_err     = mem_t[:,n] - m_hat
                            mse          = np.mean(pred_err**2)
                            response_var = np.var(mem_t[:,n])

                            if response_var > epsilon:
                                neuronal_pev[n,p,t] = 1 - mse/(response_var + epsilon)
                                neuronal_dir[n,p,t] = np.arctan2(w[2,0],w[1,0])

                    fig, ax = plt.subplots(2, figsize=[14,10])
                    im0 = ax[0].imshow(neuronal_pev[:,p,:], aspect='auto')
                    im1 = ax[1].imshow(neuronal_dir[:,p,:], aspect='auto', cmap=custom_cmap, clim=[-3,3])

                    ax[0].set_title('Neuronal PEV')
                    ax[1].set_title('Neuronal Preferred Direction')

                    fig.colorbar(im0, ax=ax[0])
                    fig.colorbar(im1, ax=ax[1])
                    fig.suptitle('Pulse {}'.format(p))

                    plt.savefig('./plots/iter{}_pev_and_dir_pulse{}.png'.format(i,p))
                    plt.clf()
                    plt.close()

                print('Diagnostics complete.\n')


            if i%par['iters_between_outputs'] == 0: #200 in list(range(len(par['trial_type']))):
                 weights = sess.run(model.var_dict)

                 results = {
                     'model_performance': model_performance,
                     'parameters': par,
                     'weights': weights,
                     'ltm_dyn_hist': np.array(ltm_dyn_hist)}
                 pickle.dump(results, open(par['save_dir'] + par['save_fn'], 'wb') )
            #     if i>=5 and all(np.array(model_performance['accuracy'][-5:]) > accuracy_threshold[acc_count]):
            #         break

            if False and i>5 and all(np.array(model_performance['accuracy'][-5:]) > accuracy_threshold[acc_count]):

                weights = sess.run(model.var_dict)
                results = {
                    'model_performance': model_performance,
                    'parameters': par,
                    'weights': weights}
                acc_str = str(int(accuracy_threshold[acc_count]*100))
                sf = save_fn[:-4] + 'acc' + acc_str + '_' + save_fn[-4:]
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

    mav = tf.placeholder(tf.float32, [1,par['n_hidden']], 'mem_avg')

    l  = tf.placeholder(tf.int32, shape=[], name='lesion')
    ci = tf.placeholder(tf.int32, shape=[], name='cut_i')
    cj = tf.placeholder(tf.int32, shape=[], name='cut_j')
    h  = tf.placeholder(tf.float32, shape=par['h_init'].shape, name='h_init')
    sx = tf.placeholder(tf.float32, shape=par['syn_x_init'].shape, name='syn_x_init')
    su = tf.placeholder(tf.float32, shape=par['syn_u_init'].shape, name='syn_u_init')

    return x, y, m, l, ci, cj, h, sx, su, mav


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            main(gpu_id=sys.argv[1])
        else:
            main()
    except KeyboardInterrupt:
        quit('\nQuit via KeyboardInterrupt.')
