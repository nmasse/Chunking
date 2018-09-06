"""
Functions used to save model data and to perform analysis
"""

import numpy as np
from parameters import *
from sklearn import svm
import time
import sys
import pickle
import stimulus
import copy
import matplotlib.pyplot as plt
from itertools import product

if len(sys.argv) > 1:
    GPU_ID = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
else:
    GPU_ID = None
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

print('gpu id:', GPU_ID)


def load_and_replace_parameters(filename, savefile=None, parameter_updates={}):

    data = pickle.load(open(filename, 'rb'))
    if savefile is None:
        data['parameters']['save_fn'] = filename + '_test.pkl'
    else:
        data['parameters']['save_fn'] = savefile

    data['parameters']['weight_load_fn'] = filename
    data['parameters']['response_multiplier'] = 1.
    data['parameters']['load_prev_weights'] = True

    data['weights']['h_init'] = data['weights']['h_init']
    #data['parameters']['dt'] = 100
    #data['parameters']['batch_train_size'] = 5


    update_parameters(data['parameters'])

    for k in parameter_updates.keys():
        par[k] = parameter_updates[k]

    #par['resp_cue_time'] = 200
    data['parameters'] = par
    return data, data['parameters']['save_fn']


def load_tensorflow_model():

    import tensorflow as tf
    import model

    # Reset graph and define all placeholders
    tf.reset_default_graph()
    x, y, m, l, ci, cj, h, sx, su = model.get_placeholders()

    # Make session, load model, and initialize weights
    sess = tf.Session()
    device = '/cpu:0'# if GPU_ID is None else '/gpu:0'
    with tf.device(device):
        mod = model.Model(x, y, m, l, ci, cj, h, sx, su)
    load_model_weights(sess)

    # Return session, model, and placeholders
    return sess, mod, x, y, m, l, ci, cj, h, sx, su


def load_model_weights(sess):
    sess.run(tf.global_variables_initializer())


def analyze_model_from_file(filename, savefile=None, analysis = False, test_mode_pulse=False, test_mode_delay=False):

    print(' --- Loading and running model for analysis.')
    results, savefile = load_and_replace_parameters(filename, savefile)
    sess, model, x, y, m, *_ = load_tensorflow_model()

    stim = stimulus.Stimulus()

    # Generate a batch of stimulus for training
    for task in par['trial_type']:
        results[task] = {}

        print('\n' + '-'*60 + '\nTask: {}\n'.format(task) + '-'*60)

        trial_info = stim.generate_trial(task, var_delay=False, var_num_pulses=False, all_RF=par['all_RF'])

        # Put together the feed dictionary
        feed_dict = {x:trial_info['neural_input'], y:trial_info['desired_output'], m:trial_info['train_mask']}

        # Run the model
        y_hat, h, syn_x, syn_u = sess.run([model.y_hat, model.hidden_hist, model.syn_x_hist, model.syn_u_hist], feed_dict=feed_dict)

        # Convert to arrays
        y_hat = np.stack(y_hat, axis=0)
        h     = np.stack(h,     axis=0)
        syn_x = np.stack(syn_x, axis=0)
        syn_u = np.stack(syn_u, axis=0)
        trial_time = np.arange(0,h.shape[0]*par['dt'], par['dt'])

        results[task]['mean_h'] = np.mean(h,axis=1)
        accuracy, pulse_accuracy = get_perf(trial_info['desired_output'], y_hat, trial_info['train_mask'], trial_info['pulse_id'])

        print('Accuracy:'.ljust(20), accuracy)
        if 'sequence' in task:
            print('Accuracy by pulse:'.ljust(20), pulse_accuracy)
        elif 'RF' in task:
            print('Accuracy by RF:'.ljust(20), pulse_accuracy)

        results[task]['task_acc'] = accuracy
        results[task]['task_pulse_acc'] = pulse_accuracy
        results[task]['task_pulse_acc_note'] = 'Accuracy by pulse.' if 'sequence' in task else 'Accuracy by RF.'
        pickle.dump(results, open(savefile, 'wb'))

        currents, tuning, simulation, decoding, cut_weight_analysis = True, True, False, True, False

        """
        Calculate currents
        """
        if currents:
            print('calculate current...')
            current_results = calculate_currents(h, syn_x, syn_u, trial_info['neural_input'], results['weights'])
            for key, val in current_results.items():
                results[task][key] = val
                #x[key] = val # added just to be able to run cut_weights in one analysis run
            pickle.dump(results, open(savefile, 'wb'))

        """
        Calculate neuronal and synaptic sample motion tuning
        """
        if tuning:
            print('calculate tuning...')
            sample = trial_info['sample']
            tuning_results = calculate_tuning(h, syn_x, syn_u, sample)
            for key, val in tuning_results.items():
                results[task][key] = val
                #x[key] = val # added just to be able to run cut_weights in one analysis run
            pickle.dump(results, open(savefile, 'wb'))

        """
        Calculate the neuronal and synaptic contributions towards solving the task
        """
        #print('weights ',results['weights']['W_in'].shape, results['weights']['W_rnn'].shape)
        if simulation:
            print('simulating network...')
            simulation_results = simulate_network(trial_info, h, syn_x, syn_u, trial_info['neural_input'], results['weights'], filename)
            for key, val in simulation_results.items():
                results[task][key] = val
            pickle.dump(results, open(savefile, 'wb'))

        """
        Decode the sample direction from neuronal activity and synaptic efficacies
        using support vector machines
        """
        if decoding:
            print('decoding activity...')
            decoding_results =  svm_wraper_simple(h, syn_x, syn_u, trial_info, num_reps = 3, num_reps_stability = 0)
            for key, val in decoding_results.items():
                print(key)
                results[task][key] = val
            print(savefile)
            pickle.dump(results, open(savefile, 'wb') )



        if cut_weight_analysis:
            print('Removing weights...')
            cut_results = cut_weights(results, trial_info, h, syn_x, syn_u, results['weights'], filename, num_reps = 1, num_top_neurons = 4)
            for key, val in cut_results.items():
                results[task][key] = val
            pickle.dump(results, open(savefile, 'wb'))
    sess.close()


def analyze_model(x, trial_info, y_hat, h, syn_x, syn_u, model_performance, weights, analysis = False, test_mode_pulse=False, pulse=0, test_mode_delay=False,stim_num=0, simulation = True, \
        cut = False, lesion = False, tuning = False, decoding = False, load_previous_file = False, save_raw_data = False):

    """
    Converts neuronal and synaptic values, stored in lists, into 3D arrays
    Creating new variable since h, syn_x, and syn_u are class members of model.py,
    and will get mofiied by functions within analysis.py
    """
    syn_x_stacked = np.stack(syn_x, axis=1)
    syn_u_stacked = np.stack(syn_u, axis=1)
    h_stacked = np.stack(h, axis=1)
    print('h_stacked', h_stacked.shape)
    trial_time = np.arange(0,h_stacked.shape[1]*par['dt'], par['dt'])
    mean_h = np.mean(np.mean(h_stacked,axis=2),axis=1)


    save_fn = par['save_dir'] + par['save_fn']

    if stim_num>0 or pulse>par['num_max_pulse']//2:
        results = pickle.load(open(save_fn, 'rb'))
    else:
        results = {
            'model_performance': model_performance,
            'parameters': par,
            'weights': weights,
            'trial_time': trial_time,
            'mean_h': mean_h,
            'timeline': trial_info['timeline']}
    #mod = model.Model(x, y, m, l, ci, cj, h, sx, su)

    pickle.dump(results, open(save_fn, 'wb') )
    print('Analysis results saved in ', save_fn)

    if save_raw_data:
        results['h'] = h
        results['syn_x'] = np.array(syn_x)
        results['syn_u'] = np.array(syn_u)
        results['y_hat'] = np.array(y_hat)
        results['trial_info'] = trial_info

    """
    Calculate accuracy after lesioning weights
    """
    if lesion:
        print('lesioning weights...')
        lesion_results = lesion_weights(trial_info, h_stacked, syn_x_stacked, syn_u_stacked, weights, trial_time)
        for key, val in lesion_results.items():
             results[key] = val
        pickle.dump(results, open(save_fn, 'wb'))

    """
    Calculate the neuronal and synaptic contributions towards solving the task
    """
    if simulation:
        print('simulating network...')
        simulation_results = simulate_network(trial_info, h_stacked, syn_x_stacked, \
            syn_u_stacked, weights)
        for key, val in simulation_results.items():
            results[key] = val
        pickle.dump(results, open(save_fn, 'wb'))

    """
    Calculate neuronal and synaptic sample motion tuning
    """
    if tuning:
        print('calculate tuning...')
        tuning_results = calculate_tuning(h_stacked, syn_x_stacked, syn_u_stacked, \
            trial_info, trial_time, weights)
        for key, val in tuning_results.items():
            results[key] = val
            x[key] = val # added just to be able to run cut_weights in one analysis run
        pickle.dump(results, open(save_fn, 'wb'))

    if cut:
        print('cutting weights...')
        cutting_results = cut_weights(x, trial_info, 0, trial_time, h_stacked, syn_x_stacked, syn_u_stacked, weights)
        for key, val in cutting_results.items():
            results[key] = val
        pickle.dump(results, open(save_fn, 'wb'))


    """
    Decode the sample direction from neuronal activity and synaptic efficacies
    using support vector machines
    """
    if decoding:
        print('decoding activity...')
        decoding_results = calculate_svms(h_stacked, syn_x_stacked, syn_u_stacked, trial_info, trial_time, \
            num_reps = par['decoding_reps'], decode_test = par['decode_test'], decode_rule = par['decode_rule'], \
            decode_sample_vs_test = par['decode_sample_vs_test'], analysis=analysis, test_mode_pulse=test_mode_pulse, pulse=pulse, test_mode_delay=test_mode_delay, stim_num=stim_num)
        for key, val in decoding_results.items():
            results[key] = val
        print('save: ', save_fn)
        pickle.dump(results, open(save_fn, 'wb') )

    #pickle.dump(results, open(save_fn, 'wb') ) -> saving after each analysis instead
    print('Analysis results saved in ', save_fn)


def calculate_svms(h, syn_x, syn_u, trial_info, trial_time, num_reps = 10, \
    decode_test = False, decode_rule = False, decode_sample_vs_test = False, analysis = False, test_mode_pulse=False, pulse=0, test_mode_delay=False, stim_num=0):

    """
    Calculates neuronal and synaptic decoding accuracies uisng support vector machines
    sample is the index of the sample motion direction for each trial_length
    rule is the rule index for each trial_length
    """



    if par['trial_type'] == 'DMC':
        """
        Will also calculate the category decoding accuracies, assuming the first half of
        the sample direction belong to category 1, and the second half belong to category 2
        """
        num_motion_dirs = len(np.unique(trial_info['sample']))
        sample = np.floor(trial_info['sample']/(num_motion_dirs/2)*np.ones_like(trial_info['sample']))
        test = np.floor(trial_info['test']/(num_motion_dirs/2)*np.ones_like(trial_info['sample']))
        rule = trial_info['rule']
    elif par['trial_type'] == 'dualDMS':
        sample = trial_info['sample']
        rule = trial_info['rule'][:,0] + 2*trial_info['rule'][:,1]
        par['num_rules'] = 4
    elif par['trial_type'] == 'DMS+DMC':
        # rule 0 is DMS, rule 1 is DMC
        ind_rule = np.where(trial_info['rule']==1)[0]
        num_motion_dirs = len(np.unique(trial_info['sample']))
        sample = np.array(trial_info['sample'])
        test = np.array(trial_info['test'])
        # change DMC sample motion directions into categories
        sample[ind_rule] = np.floor(trial_info['sample'][ind_rule]/(num_motion_dirs/2)*np.ones_like(trial_info['sample'][ind_rule]))
        test[ind_rule] = np.floor(trial_info['test'][ind_rule]/(num_motion_dirs/2)*np.ones_like(trial_info['sample'][ind_rule]))
        rule = trial_info['rule']

    else:
        sample = np.array(trial_info['sample'])
        rule = np.array(trial_info['rule'])
        print('sample ', sample.shape)

    # if trial_info['test'].ndim == 2:
    #     test = trial_info['test'][:,0]
    # else:
    #     test = np.array(trial_info['test'])


    print('sample decoding...num_reps = ', num_reps)

    if analysis:
        decoding_results['neuronal_sample_decoding'+str(stim_num)], decoding_results['synaptic_sample_decoding'+str(stim_num)],decoding_results['combined_decoding'+str(stim_num)] = \
            svm_wraper(lin_clf, h, syn_efficacy, combined, sample, rule, num_reps, trial_time,analysis, test_mode_pulse, pulse, stim_num)
        # neu, syn, comb = svm_wraper(lin_clf, h, syn_efficacy, combined, sample, rule, num_reps, trial_time, analysis, stim_num)
        # decoding_results['neuronal_sample_decoding'] = np.concatenate((decoding_results['neuronal_sample_decoding'], neu), axis = 1)
        # decoding_results['synaptic_sample_decoding'] = np.concatenate((decoding_results['synaptic_sample_decoding'], syn), axis = 1)
        # decoding_results['combined_decoding'] = np.concatenate((decoding_results['combined_decoding'], comb), axis = 1)
    elif test_mode_pulse:
        decoding_results['neuronal_sample_decoding'+str(pulse)], decoding_results['synaptic_sample_decoding'+str(pulse)],decoding_results['combined_decoding'+str(pulse)] = \
            svm_wraper(lin_clf, h, syn_efficacy, combined, sample, rule, num_reps, trial_time,analysis, test_mode_pulse, pulse)
    elif test_mode_delay:
        decoding_results['neuronal_sample_decoding'], decoding_results['synaptic_sample_decoding'],decoding_results['combined_decoding'] = \
            svm_wraper(lin_clf, h, syn_efficacy, combined, sample, rule, num_reps, trial_time,analysis, test_mode_pulse, pulse, test_mode_delay)
    else:
        decoding_results['neuronal_sample_decoding'], decoding_results['synaptic_sample_decoding'],decoding_results['combined_decoding'] = \
            svm_wraper(lin_clf, h, syn_efficacy, combined, sample, rule, num_reps, trial_time)

    if decode_sample_vs_test:
        print('sample vs. test decoding...')
        decoding_results['neuronal_sample_test_decoding'], decoding_results['synaptic_sample_test_decoding'] = \
            svm_wraper_sample_vs_test(lin_clf, h, syn_efficacy, trial_info['sample'], trial_info['test'], num_reps, trial_time)

    if decode_test:
        print('test decoding...')
        decoding_results['neuronal_test_decoding'], decoding_results['synaptic_test_decoding'] = \
            svm_wraper(lin_clf, h, syn_efficacy, test, rule, num_reps, trial_time)

    if decode_rule:
        print('rule decoding...')
        decoding_results['neuronal_rule_decoding'], decoding_results['synaptic_rule_decoding'] = \
            svm_wraper(lin_clf, h, syn_efficacy, trial_info['rule'], np.zeros_like(sample), num_reps, trial_time)

    return decoding_results


def svm_wraper_simple(h, syn_x, syn_u, trial_info, num_reps = 3, num_reps_stability = 0):

    par['decode_stability'] = False
    train_pct = 0.5
    num_time_steps, num_trials, _ = h.shape
    lin_clf = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr', shrinking=False, tol=1e-3)

    score = np.zeros((3, par['num_pulses'], num_reps, num_time_steps), dtype = np.float32)
    score_dynamic = np.zeros((3, par['num_pulses'], num_reps_stability, num_time_steps, num_time_steps), dtype = np.float32)

    # number of reps used to calculate encoding stability should not be larger than number of normal deocding reps
    num_reps_stability = np.minimum(num_reps_stability, num_reps)

    for p in range(par['num_pulses']):
        print('Decoding pulse number ', p)
        labels = trial_info['sample'][:, p]
        for rep in range(num_reps):

            q = np.random.permutation(num_trials)
            ind_train = q[:round(train_pct*num_trials)]
            ind_test = q[round(train_pct*num_trials):]

            for data_type in [0,1,2]:
                if data_type == 0:
                    z = np.array(h)
                elif data_type == 1:
                    z = np.array(syn_x*syn_u)
                elif data_type == 2:
                    z = np.array( np.concatenate((h, syn_x*syn_u), axis=2))

                for t in range(num_time_steps):
                    lin_clf.fit(z[t,ind_train,:], np.array(labels[ind_train]))
                    predicted_sample = lin_clf.predict(z[t,ind_test,:])
                    score[data_type, p, rep, t] = np.mean( labels[ind_test]==predicted_sample)

                    if rep < num_reps_stability and par['decode_stability']:
                        print('Should be see this.')
                        for t1 in range(num_time_steps):
                            predicted_sample = lin_clf.predict(z[t1,ind_test,:].T)
                            score_dynamic[data_type, p, rep, t, t1] = np.mean(labels[ind_test]==predicted_sample)

    print('decoding done')
    results = {'neuronal_decoding': score[0,:,:,:], 'synaptic_decoding': score[1,:,:,:], 'combined_decoding': score[2,:,:,:]}
    return results


def svm_wraper(lin_clf, h, syn_eff, combo, stim, rule, num_reps, trial_time, analysis=False, test_mode_pulse=False, pulse=0, test_mode_delay=False,stim_num=0):

    """
    Wraper function used to decode sample/test or rule information
    from hidden activity (h) and synaptic efficacies (syn_eff)
    """
    train_pct = 0.75
    trials_per_cond = 25
    num_time_steps, num_trials, _ = h.shape
    num_rules = len(np.unique(rule))

    if sequence in par['trial_type']:
        num_stim = par['num_pulses']
    else:
        num_stim = par['num_RFs']

    #num_stim = par['num_pulses'] if par['trial_type']=='chunking' else par['num_receptive_fields']

    score_h = np.zeros((num_rules, num_stim, num_reps, num_time_steps), dtype = np.float32)
    score_syn_eff = np.zeros((num_rules, num_stim, num_reps, num_time_steps), dtype = np.float32)
    score_combo = np.zeros((num_rules, num_stim, num_reps, num_time_steps), dtype = np.float32)

    for r in range(num_rules):
        ind_rule = np.where(rule==r)[1]

        for n in range(num_stim):
            if 'sequence' or 'RF' in par['trial_type']:
                current_stim = stim[:,n]
            else:
                current_stim = np.array(stim)

            num_unique_stim = len(np.unique(stim[ind_rule]))
            if num_unique_stim <= 2:
                trials_per_stim = 100
            else:
                trials_per_stim = 25
            print('Rule ', r, ' num conds ', num_unique_stim)

            equal_train_ind = np.zeros((num_unique_stim*trials_per_cond), dtype = np.uint16)
            equal_test_ind = np.zeros((num_unique_stim*trials_per_cond), dtype = np.uint16)

            stim_ind = []
            for c in range(num_unique_stim):
                stim_ind.append(ind_rule[np.where(current_stim[ind_rule] == c)[0]])
                if len(stim_ind[c]) < 4:
                    print('Not enough trials for this stimulus!')
                    print('Setting cond_ind to [0,1,2,3]')
                    stim_ind[c] = [0,1,2,3]

            for rep in range(num_reps):
                for c in range(num_unique_stim):
                    u = range(c*trials_per_cond, (c+1)*trials_per_stim)
                    q = np.random.permutation(len(stim_ind[c]))
                    i = int(np.round(len(stim_ind[c])*train_pct))
                    train_ind = stim_ind[c][q[:i]]
                    test_ind = stim_ind[c][q[i:]]

                    q = np.random.randint(len(train_ind), size = trials_per_stim)
                    equal_train_ind[u] =  train_ind[q]
                    q = np.random.randint(len(test_ind), size = trials_per_stim)
                    equal_test_ind[u] =  test_ind[q]

                for t in range(num_time_steps):
                    if trial_time[t] <= par['dead_time']:
                        # no need to analyze activity during dead time
                        continue

                    score_h[r,n,rep,t] = calc_svm(lin_clf, h[:,t,:].T, current_stim, current_stim, equal_train_ind, equal_test_ind)
                    score_syn_eff[r,n,rep,t] = calc_svm(lin_clf, syn_eff[:,t,:].T, current_stim, current_stim, equal_train_ind, equal_test_ind)
                    score_combo[r,n,rep,t] = calc_svm(lin_clf, combo[:,t,:].T, current_stim, current_stim, equal_train_ind, equal_test_ind)


    return score_h, score_syn_eff, score_combo


def calc_svm(lin_clf, y, train_conds, test_conds, train_ind, test_ind):

    n_test_inds = len(test_ind)
    # normalize values between 0 and 1
    for i in range(y.shape[1]):
        m1 = y[train_ind,i].min()
        m2 = y[train_ind,i].max()
        y[:,i] -= m1
        if m2>m1:
            if par['svm_normalize']:
                y[:,i] /=(m2-m1)

    lin_clf.fit(y[train_ind,:], train_conds[train_ind])
    pred_stim = lin_clf.predict(y[test_ind,:])
    score = np.mean(test_conds[test_ind]==pred_stim)

    return score


def lesion_weights(trial_info, h, syn_x, syn_u, network_weights, trial_time):

    lesion_results = {'lesion_accuracy_rnn': np.ones((par['num_rules'], par['n_hidden'],par['n_hidden']), dtype=np.float32),
                      'lesion_accuracy_out': np.ones((par['num_rules'], 3,par['n_hidden']), dtype=np.float32)}

    for r in range(par['num_rules']):
        trial_ind = np.where(trial_info['rule']==r)[0]
        # network inputs/outputs
        test_onset = (par['dead_time']+par['fix_time'])//par['dt']
        x = np.split(trial_info['neural_input'][:,test_onset:,trial_ind],len(trial_time)-test_onset,axis=1)
        y = np.array(trial_info['desired_output'][:,test_onset:,trial_ind])
        train_mask = np.array(trial_info['train_mask'][test_onset:,trial_ind])
        hidden_init = h[:,test_onset-1,trial_ind]
        syn_x_init = syn_x[:,test_onset-1,trial_ind]
        syn_u_init = syn_u[:,test_onset-1,trial_ind]

        test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']

        hidden_init_test = h[:,test_onset-1,trial_ind]
        syn_x_init_test = syn_x[:,test_onset-1,trial_ind]
        syn_u_init_test = syn_u[:,test_onset-1,trial_ind]
        x_test = np.split(trial_info['neural_input'][:,test_onset:,trial_ind],len(trial_time)-test_onset,axis=1)
        y_test = trial_info['desired_output'][:,test_onset:,trial_ind]
        train_mask_test = trial_info['train_mask'][test_onset:,trial_ind]

        print('Lesioning output weights...')
        for n1 in range(3):
            for n2 in range(par['n_hidden']):

                if network_weights['W_out'][n1,n2] <= 0:
                    continue

                # create new dict of weights
                weights_new = {}
                for k,v in network_weights.items():
                    weights_new[k] = np.array(v+1e-32)

                # lesion weights
                q = np.ones((3,par['n_hidden']), dtype=np.float32)
                q[n1,n2] = 0
                weights_new['W_out'] *= q

                # simulate network
                y_hat, _, _, _ = run_model(x_test, hidden_init_test, syn_x_init_test, syn_u_init_test, weights_new)
                lesion_results['lesion_accuracy_out'][r,n1,n2],_,_ = get_perf(y_test, y_hat, train_mask_test)

        print('Lesioning recurrent weights...')
        for n1 in range(par['n_hidden']):
            for n2 in range(par['n_hidden']):

                if network_weights['W_rnn'][n1,n2] <= 0:
                    continue

                weights_new = {}
                for k,v in network_weights.items():
                    weights_new[k] = np.array(v+1e-32)

                # lesion weights
                q = np.ones((par['n_hidden'],par['n_hidden']), dtype=np.float32)
                q[n1,n2] = 0
                weights_new['W_rnn'] *= q

                # simulate network
                y_hat, hidden_state_hist, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, weights_new)
                lesion_results['lesion_accuracy_rnn'][r,n1,n2],_,_ = get_perf(y, y_hat, train_mask)

                #y_hat, _, _, _ = run_model(x_test, hidden_init_test, syn_x_init_test, syn_u_init_test, weights_new)
                #lesion_results['lesion_accuracy_rnn_test'][r,n1,n2],_,_ = get_perf(y_test, y_hat, train_mask_test)

                """
                if accuracy_rnn_start[n1,n2] < -1:

                    h_stacked = np.stack(hidden_state_hist, axis=1)

                    neuronal_decoding[n1,n2,:,:,:], _ = calculate_svms(h_stacked, syn_x, syn_u, trial_info['sample'], \
                        trial_info['rule'], trial_info['match'], trial_time, num_reps = num_reps)

                    neuronal_pref_dir[n1,n2,:,:], neuronal_pev[n1,n2,:,:], _, _ = calculate_sample_tuning(h_stacked, \
                        syn_x, syn_u, trial_info['sample'], trial_info['rule'], trial_info['match'], trial_time)
                """


    return lesion_results


def simulate_network(trial_info, h, syn_x, syn_u, network_input, network_weights, filename, num_reps = 1):

    test_length = int(par['resp_cue_time']//par['dt'])
    updates = {'num_time_steps' : test_length}
    results = load_and_replace_parameters(filename, parameter_updates=updates)
    sess, model, x, y, ma, l, ci, cj, hi, sx, su = load_tensorflow_model()

    """
    Simulation will start from the start of the test period until the end of trial
    """
    trial_length, batch_train_size, _ = h.shape
    mean_pulse_id = np.mean(trial_info['pulse_id'], axis = 1) # trial_info['pulse_id'] should be identical across all trials
    pulse_onsets = [np.min(np.where(mean_pulse_id==i)[0]) for i in range(par['num_pulses'])]
    test_length = par['resp_cue_time']//par['dt']

    simulation_results = {
        'accuracy_no_shuffle'           : np.zeros((par['num_pulses'], par['n_hidden'], num_reps)),
        'accuracy_neural_shuffled'      : np.zeros((par['num_pulses'], par['n_hidden'], num_reps)),
        'accuracy_syn_shuffled'         : np.zeros((par['num_pulses'], par['n_hidden'], num_reps))}

    for p in range(par['num_pulses']):

        test_onset = pulse_onsets[p]
        print(pulse_onsets, test_onset,test_length)

        x_input = trial_info['neural_input'][test_onset:test_onset+test_length,:,:]
        y_target = trial_info['desired_output'][test_onset:test_onset+test_length,:,:]
        train_mask = trial_info['train_mask'][test_onset:test_onset+test_length, :]
        pulse_id = trial_info['pulse_id'][test_onset:test_onset+test_length, :]


        for n in range(num_reps):
            print('pulse ', p, ' rep ', n, "out of ", num_reps)

            """
            Calculating behavioral accuracy without shuffling
            """
            hidden_init = np.array(h[test_onset-1,:,:])
            syn_x_init = np.array(syn_x[test_onset-1,:,:])
            syn_u_init = np.array(syn_u[test_onset-1,:,:])

            sess.run([model.load_h_init, model.load_syn_x_init, model.load_syn_u_init], \
                feed_dict={hi:hidden_init, sx:syn_x_init, su:syn_u_init})
            y_hat = np.stack(sess.run(model.y_hat, feed_dict={x:x_input, y:y_target, ma:train_mask}), axis=0)

            simulation_results['accuracy_no_shuffle'][p,:,n], _ = get_perf(y_target, y_hat, train_mask, pulse_id)

            ind_shuffle = np.random.permutation(batch_train_size)


            for m in range(par['n_hidden']):
                """
                Keep the synaptic values fixed, permute the neural activity
                """

                hidden_init = np.array(h[test_onset-1,:,:])
                syn_x_init = np.array(syn_x[test_onset-1,:,:])
                syn_u_init = np.array(syn_u[test_onset-1,:,:])
                hidden_init[:,m] = hidden_init[ind_shuffle,m]

                sess.run([model.load_h_init, model.load_syn_x_init, model.load_syn_u_init], \
                    feed_dict={hi:hidden_init, sx:syn_x_init, su:syn_u_init})
                y_hat = np.stack(sess.run(model.y_hat, feed_dict={x:x_input, y:y_target, ma:train_mask}), axis=0)

                simulation_results['accuracy_neural_shuffled'][p,m,n], _ = get_perf(y_target, y_hat, train_mask, pulse_id)
                acc, _  = get_perf(y_target, y_hat, train_mask, pulse_id)

                """
                Keep the hidden values fixed, permute synaptic values
                """
                hidden_init = np.array(h[test_onset-1,:,:])
                syn_x_init = np.array(syn_x[test_onset-1,:,:])
                syn_u_init = np.array(syn_u[test_onset-1,:,:])
                syn_x_init[:,m] = syn_x_init[ind_shuffle,m]
                syn_u_init[:,m] = syn_u_init[ind_shuffle,m]

                sess.run([model.load_h_init, model.load_syn_x_init, model.load_syn_u_init], \
                    feed_dict={hi:hidden_init, sx:syn_x_init, su:syn_u_init})
                y_hat = np.stack(sess.run(model.y_hat, feed_dict={x:x_input, y:y_target, ma:train_mask}), axis=0)

                simulation_results['accuracy_syn_shuffled'][p,m,n], _ = get_perf(y_target, y_hat, train_mask, pulse_id)

    return simulation_results


def calculate_currents(h, syn_x, syn_u, network_input, network_weights):

    trial_length = h.shape[0]
    current_results = {
        'exc_current'            :  np.zeros((trial_length, par['n_hidden'], 2),dtype=np.float32),
        'inh_current'            :  np.zeros((trial_length, par['n_hidden'], 2),dtype=np.float32),
        'rnn_current'            :  np.zeros((trial_length, par['n_hidden'], 2),dtype=np.float32),
        'motion_current'         :  np.zeros((trial_length, par['n_hidden']),dtype=np.float32),
        'fix_current'            :  np.zeros((trial_length, par['n_hidden']),dtype=np.float32),
        'cue_current'            :  np.zeros((trial_length, par['n_hidden']),dtype=np.float32)}

    mean_activity     = np.mean(h, axis=1)
    mean_eff_activity = np.mean(h*syn_x*syn_u, axis=1)
    input_activity    = np.mean(network_input, axis=1)

    mot  = par['total_motion_tuned']
    fix  = par['total_motion_tuned'] + par['num_fix_tuned']
    cue  = par['total_motion_tuned'] + par['num_fix_tuned'] + par['num_cue_tuned']
    rule = par['total_motion_tuned'] + par['num_fix_tuned'] + par['num_cue_tuned'] + par['num_rule_tuned']

    motion_rng  = range(mot)
    fix_rng     = range(mot, fix)
    cue_rng     = range(fix, cue)
    rule_rng    = range(cue, rule)

    ei_index = par['num_exc_units']

    current_results['exc_current'][:, :, 0] = mean_activity[:,:ei_index] @ network_weights['W_rnn'][:ei_index,:]
    current_results['exc_current'][:, :, 1] = mean_eff_activity[:,:ei_index] @ network_weights['W_rnn'][:ei_index,:]
    current_results['inh_current'][:, :, 0] = mean_activity[:,ei_index:] @ network_weights['W_rnn'][ei_index:,:]
    current_results['inh_current'][:, :, 1] = mean_eff_activity[:,ei_index:] @ network_weights['W_rnn'][ei_index:,:]

    current_results['motion_current'] = input_activity[:,motion_rng] @ network_weights['W_in'][motion_rng,:]
    current_results['fix_current']    = input_activity[:,fix_rng] @ network_weights['W_in'][fix_rng,:]
    current_results['cue_current']    = input_activity[:,cue_rng] @ network_weights['W_in'][cue_rng,:]

    for t in range(trial_length):
        current_results['rnn_current'][t, :, 0] = mean_activity[t,:] @ network_weights['W_rnn']
        current_results['rnn_current'][t, :, 1] = mean_eff_activity[t,:] @ network_weights['W_rnn']

    return current_results


def cut_weights(results, trial_info, h, syn_x, syn_u, network_weights, filename, num_reps = 1, num_top_neurons = 1):

    trial_length = h.shape[0]

    cutting_results = {
        'cut_neurons'             : np.zeros((par['num_pulses'], num_top_neurons),dtype=np.float32),
        'accuracy_before_cut'     : np.zeros((par['num_pulses'], par['num_pulses']),dtype=np.float32),
        'accuracy_after_cut_start': np.zeros((par['num_pulses'], par['num_pulses']),dtype=np.float32),
        'accuracy_after_cut_delay': np.zeros((par['num_pulses'], par['num_pulses']),dtype=np.float32),
        'synaptic_pev_after_cut'          : np.zeros((par['n_hidden'], par['num_pulses'], trial_length),dtype=np.float32),
        'neuronal_pev_after_cut'          : np.zeros((par['n_hidden'], par['num_pulses'], trial_length),dtype=np.float32),
        'neuronal_pref_dir_after_cut'     : np.zeros((par['n_hidden'],  par['num_pulses'], trial_length), dtype=np.float32),
        'synaptic_pref_dir_after_cut'     : np.zeros((par['n_hidden'],  par['num_pulses'], trial_length), dtype=np.float32)}

    # Determine where fixation ends, so as to start the model from there.
    mean_fixation = np.mean(trial_info['desired_output'][:,:,0], axis=1)
    cut_start = np.where(mean_fixation == 0.)[0][0] - 1
    print('Cut weight trials start at time step {}.'.format(cut_start))

    updates = {'num_time_steps' : trial_length - cut_start}
    results = load_and_replace_parameters(filename, parameter_updates=updates)
    sess, model, x, y, ma, l, ci, cj, hi, sx, su = load_tensorflow_model()

    for p in range(par['num_pulses']):
        print(p, "out of ", par['num_pulses'], " pulses")

        h_init = np.array(h[0,:,:])
        syn_x_init = np.array(syn_x[0,:,:])
        syn_u_init = np.array(syn_u[0,:,:])

        h_init_delay = np.array(h[cut_start-1,:,:])
        syn_x_init_delay = np.array(syn_x[cut_start-1,:,:])
        syn_u_init_delay = np.array(syn_u[cut_start-1,:,:])

        """
        Calculating behavioral accuracy without shuffling
        """

        x_input    = trial_info['neural_input'][cut_start:,:,:]
        y_target   = trial_info['desired_output'][cut_start:,:,:]
        train_mask = trial_info['train_mask'][cut_start:,:]

        sess.run([model.load_h_init, model.load_syn_x_init, model.load_syn_u_init], \
            feed_dict={hi:h_init_delay, sx:syn_x_init_delay, su:syn_u_init_delay})
        y_hat = np.stack(sess.run(model.y_hat, feed_dict={x:x_input, y:y_target, ma:train_mask}), axis=0)

        _,pulse_acc =  get_perf(trial_info['desired_output'][1:,:,:], y_hat, \
            trial_info['train_mask'][1:,:], trial_info['pulse_id'][1:,:])
        cutting_results['accuracy_before_cut'][p,:] = pulse_acc

        """
        Cutting top neurons from synaptic_pev result
        """
        ind = np.argsort(results['synaptic_pev'][:, p, cut_start])
        top_neurons = ind[-num_top_neurons:]
        cutting_results['cut_neurons'][p,:] = top_neurons

        current_weights = copy.deepcopy(network_weights)
        #current_weights = {**network_weights}


        for i,j in product(top_neurons,top_neurons):
            current_weights['W_rnn'][i, j] = 0


        h_init = np.array(h[0,:,:])
        syn_x_init = np.array(syn_x[0,:,:])
        syn_u_init = np.array(syn_u[0,:,:])

        y_hat_cut, h_cut, syn_x_cut, syn_u_cut = run_model(x, h_init, syn_x_init, syn_u_init, current_weights)
        _,pulse_acc_start =  get_perf(trial_info['desired_output'][1:,:,:], y_hat_cut, \
            trial_info['train_mask'][1:,:], trial_info['pulse_id'][1:,:])
        cutting_results['accuracy_after_cut_start'][p,:] = pulse_acc_start

        print('ds ', cutting_results['synaptic_pev_after_cut'][:, p, 1:].shape)
        sample = np.reshape(trial_info['sample'][:, p], (-1,1))
        tuning_results_cut = calculate_tuning(h_cut, syn_x_cut, syn_u_cut, sample)
        cutting_results['synaptic_pev_after_cut'][:, p, 1:] = np.squeeze(tuning_results_cut['synaptic_pev'])
        cutting_results['synaptic_pref_dir_after_cut'][:, p, 1:] = np.squeeze(tuning_results_cut['synaptic_pref_dir'])
        cutting_results['neuronal_pev_after_cut'][:, p, 1:] = np.squeeze(tuning_results_cut['neuronal_pev'])
        cutting_results['neuronal_pref_dir_after_cut'][:, p, 1:] = np.squeeze(tuning_results_cut['neuronal_pref_dir'])


        # Apply cutting instead of inits
        #sess.run([model.load_h_init, model.load_syn_x_init, model.load_syn_u_init], \
        #    feed_dict={hi:hidden_init, sx:syn_x_init, su:syn_u_init})
        y_hat = np.stack(sess.run(model.y_hat, feed_dict={x:x_input, y:y_target, ma:train_mask}), axis=0)

        #y_hat_cut, _, _, _ = run_model(x_delay, h_init_delay, syn_x_init_delay, syn_u_init_delay, current_weights)
        _, pulse_acc_delay = get_perf(trial_info['desired_output'][cut_start:,:,:], y_hat_cut, trial_info['train_mask'][cut_start:,:], \
            trial_info['pulse_id'][cut_start:,:])
        cutting_results['accuracy_after_cut_delay'][p,:] = pulse_acc_delay



    return cutting_results


def calculate_tuning(h, syn_x, syn_u, sample):

    epsilon = 1e-9
    """
    Calculates neuronal and synaptic sample motion direction tuning
    """
    num_time_steps = h.shape[0]
    num_pulses = sample.shape[1]

    print('pulses, time ',num_pulses,num_time_steps)

    # want zeros(n_hidden, n_pulse, n_time)

    tuning_results = {
        'neuronal_pref_dir'     : np.zeros((par['n_hidden'],  num_pulses, num_time_steps), dtype=np.float32),
        'synaptic_pref_dir'     : np.zeros((par['n_hidden'],  num_pulses, num_time_steps), dtype=np.float32),
        'neuronal_pev'          : np.zeros((par['n_hidden'],  num_pulses, num_time_steps), dtype=np.float32),
        'synaptic_pev'          : np.zeros((par['n_hidden'],  num_pulses, num_time_steps), dtype=np.float32),
        'pulse_accuracy_tuning' : np.zeros((par['num_pulses'], num_pulses), dtype=np.float32)}


    """
    The synaptic efficacy is the product of syn_x and syn_u, will decode sample
    direction from this value
    """
    syn_efficacy = syn_x*syn_u

    sample_dir = np.ones((par['batch_train_size'], 3, par['num_pulses']))


    for i in range(num_pulses):
        sample_dir[:,1, i] = np.cos(2*np.pi*sample[:,i]/par['num_motion_dirs'])
        sample_dir[:,2, i] = np.sin(2*np.pi*sample[:,i]/par['num_motion_dirs'])

        for n in range(par['n_hidden']):
            for t in range(num_time_steps):

                # Neuronal sample tuning
                w = np.linalg.lstsq(sample_dir[:,:,i], h[t,:,n], rcond=None)
                w = w[0][...,np.newaxis]
                h_hat =  np.dot(sample_dir[:,:,i], w).T
                pred_err = h[t,:,n] - h_hat
                mse = np.mean(pred_err**2) # var (h-h_hat)
                response_var = np.var(h[t,:,n]) # var(h)

                if response_var > epsilon:
                    tuning_results['neuronal_pev'][n,i,t] = 1 - mse/(response_var + epsilon)
                    tuning_results['neuronal_pref_dir'][n,i,t] = np.arctan2(w[2,0],w[1,0])

                # Synaptic sample tuning
                w = np.linalg.lstsq(sample_dir[:,:,i], syn_efficacy[t,:,n], rcond=None)
                w = w[0][...,np.newaxis]
                syn_hat = np.dot(sample_dir[:,:,i], w).T
                pred_err = syn_efficacy[t,:,n] - syn_hat
                mse = np.mean(pred_err**2)
                response_var = np.var(syn_efficacy[t,:,n])

                if response_var > epsilon:
                    tuning_results['synaptic_pev'][n,i,t] = 1 - mse/(response_var + epsilon)
                    tuning_results['synaptic_pref_dir'][n,i,t] = np.arctan2(w[2,0],w[1,0])

    return tuning_results


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
        W_rnn_effective = np.dot(np.maximum(0,weights['W_rnn']), par['EI_matrix'])
    else:
        W_rnn_effective = weights['W_rnn']


    """
    Update the synaptic plasticity paramaters
    """
    if par['synapse_config'] == 'std_stf':
        # implement both synaptic short term facilitation and depression
        """
        syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
        syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
        syn_x = np.minimum(1, np.maximum(0, syn_x))
        syn_u = np.minimum(1, np.maximum(0, syn_u))
        h_post = syn_u*syn_x*h
        """
        syn_x1 = syn_x + par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
        syn_u1 = syn_u + par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
        syn_x1 = np.minimum(1, np.maximum(0, syn_x1))
        syn_u1 = np.minimum(1, np.maximum(0, syn_u1))
        h_post = syn_u1*syn_x1*h

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
                   + par['alpha_neuron']*(np.dot(np.maximum(0,weights['W_in']), np.transpose(np.maximum(0, rnn_input)))
                   + np.dot(W_rnn_effective, h_post) + weights['b_rnn'])
                   + np.random.normal(0, par['noise_rnn'],size=(h.shape[1],par['n_hidden'])))

    h *= suppress_activity

    return h, syn_x1, syn_u1


def get_perf(y, y_hat, mask, pulse_id):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when y[0,:,:] is not 0
    y is the desired output
    y_hat is the actual output
    """

    """y_hat = np.stack(y_hat)

    fig, ax = plt.subplots(3)
    ax[0].imshow(y[:,0,:], aspect='auto')
    ax[1].imshow(y_hat[:,0,:], aspect='auto')
    ax[2].imshow(mask[:,0:1], aspect='auto')

    plt.show()
    quit()"""

    #print("Entering get_perf...")
    #print(np.sum(mask))
    y_hat = np.stack(y_hat, axis=0)
    mask_test = mask*(y[:,:,0]==0)
    y_max = np.argmax(y, axis = 2)
    y_hat = np.argmax(y_hat, axis = 2)
    accuracy = np.sum(np.float32(y_max == y_hat)*mask_test)/np.sum(mask_test)

    pulse_accuracy = np.zeros((par['num_pulses']))
    for i in range(par['num_pulses']):
        current_mask = mask_test*(pulse_id == i)
        pulse_accuracy[i] = np.sum(np.float32(y_max == y_hat)*current_mask)/np.sum(current_mask)

    return accuracy, pulse_accuracy


def get_coord_perf(target, output, mask, pulse_id):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when target[:,:,-1] is not 0
    """

    output = np.stack(output)

    output = np.swapaxes(np.array(output),1,2)
    mask_test = mask*(1-((target[:,:,0]==0) * (target[:,:,1]==0)))

    accuracy =  np.sum(mask_test*np.float32((np.absolute(target[:,:,0] - output[:,:,0]) < par['tol']) * (np.absolute(target[:,:,1] - output[:,:,1]) < par['tol'])))/np.sum(mask_test)

    pulse_accuracy = np.zeros((par['num_pulses']))
    for i in range(par['num_pulses']):
        current_mask = mask_test*(pulse_id == i)
        pulse_accuracy[i] = np.sum(current_mask*np.float32((np.absolute(target[:,:,0] - output[:,:,0]) < par['tol']) * (np.absolute(target[:,:,1] - output[:,:,1]) < par['tol'])))/np.sum(current_mask)

    return accuracy, pulse_accuracy
