"""
Functions used to save model data and to perform analysis
"""

import numpy as np
from parameters import *
from sklearn import svm
import time
import pickle
import stimulus
import matplotlib.pyplot as plt
import copy

def analyze_model_from_file(filename, savefile = None, analysis = False, test_mode_pulse=False, test_mode_delay=False):

    x = pickle.load(open('./savedir/'+filename, 'rb'))
    if savefile is None:
        x['parameters']['save_fn'] = 'test.pkl'
    else:
        x['parameters']['save_fn'] = savefile
    update_parameters(x['parameters'])
    stim = stimulus.Stimulus()
    if analysis:
        for i in range(x['parameters']['num_pulses']):
            trial_info = stim.generate_trial(analysis = True, num_fixed =i)
            input_data = np.squeeze(np.split(trial_info['neural_input'], x['parameters']['num_time_steps'], axis=1))

            y_hat, h, syn_x, syn_u = run_model(input_data, x['parameters']['h_init'], \
                x['parameters']['syn_x_init'], x['parameters']['syn_u_init'], x['weights'])

            h = np.squeeze(np.split(h, x['parameters']['num_time_steps'], axis=1))
            syn_x = np.squeeze(np.split(syn_x, x['parameters']['num_time_steps'], axis=1))
            syn_u = np.squeeze(np.split(syn_u, x['parameters']['num_time_steps'], axis=1))

            analyze_model(x,trial_info, y_hat, h, syn_x, syn_u, None, x['weights'], analysis = True, stim_num = i, simulation = False, cut = True,\
                    lesion = False, tuning = False, decoding = True, load_previous_file = False, save_raw_data = False)
    elif test_mode_pulse:
        for i in range(x['parameters']['num_max_pulse']//2,x['parameters']['num_max_pulse']+1):
            trial_info = stim.generate_trial(analysis = False, num_fixed =0,var_delay=x['parameters']['var_delay'],var_resp_delay=x['parameters']['var_resp_delay'],var_num_pulses=x['parameters']['var_num_pulses'],test_mode_pulse=True,pulse=i)
            input_data = np.squeeze(np.split(trial_info['neural_input'], x['parameters']['num_time_steps'], axis=1))

            y_hat, h, syn_x, syn_u = run_model(input_data, x['parameters']['h_init'], \
                x['parameters']['syn_x_init'], x['parameters']['syn_u_init'], x['weights'])

            h = np.squeeze(np.split(h, x['parameters']['num_time_steps'], axis=1))
            syn_x = np.squeeze(np.split(syn_x, x['parameters']['num_time_steps'], axis=1))
            syn_u = np.squeeze(np.split(syn_u, x['parameters']['num_time_steps'], axis=1))

            analyze_model(x,trial_info, y_hat, h, syn_x, syn_u, None, x['weights'], analysis = False, test_mode_pulse=True, pulse = i, simulation = True, cut = True,\
                    lesion = False, tuning = True, decoding = True, load_previous_file = False, save_raw_data = False)
    elif test_mode_delay:
        trial_info = stim.generate_trial(analysis = False,num_fixed=0,var_delay=x['parameters']['var_delay'],var_resp_delay=x['parameters']['var_resp_delay'],var_num_pulses=x['parameters']['var_num_pulses'],test_mode_pulse=test_mode_pulse,test_mode_delay=test_mode_delay)
        input_data = np.squeeze(np.split(trial_info['neural_input'], x['parameters']['num_time_steps'], axis=1))

        y_hat, h, syn_x, syn_u = run_model(input_data, x['parameters']['h_init'], \
            x['parameters']['syn_x_init'], x['parameters']['syn_u_init'], x['weights'])

        h = np.squeeze(np.split(h, x['parameters']['num_time_steps'], axis=1))
        syn_x = np.squeeze(np.split(syn_x, x['parameters']['num_time_steps'], axis=1))
        syn_u = np.squeeze(np.split(syn_u, x['parameters']['num_time_steps'], axis=1))
        analyze_model(x, trial_info, y_hat, h, syn_x, syn_u, None, x['weights'], simulation = True, cut = True,\
                lesion = False, tuning = True, decoding = True, load_previous_file = False, save_raw_data = False)
    else:
        trial_info = stim.generate_trial()
        print(trial_info['neural_input'].shape)
        print(x['parameters']['num_time_steps'])
        input_data = np.squeeze(np.split(trial_info['neural_input'], x['parameters']['num_time_steps'], axis=1))

        y_hat, h, syn_x, syn_u = run_model(input_data, x['parameters']['h_init'], \
            x['parameters']['syn_x_init'], x['parameters']['syn_u_init'], x['weights'])

        h = np.squeeze(np.split(h, x['parameters']['num_time_steps'], axis=1))
        syn_x = np.squeeze(np.split(syn_x, x['parameters']['num_time_steps'], axis=1))
        syn_u = np.squeeze(np.split(syn_u, x['parameters']['num_time_steps'], axis=1))
        analyze_model(x, trial_info, y_hat, h, syn_x, syn_u, None, x['weights'], simulation = False, cut = True,\
                lesion = False, tuning = False, decoding = False, load_previous_file = False, save_raw_data = False)


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
        pickle.dump(results, open(save_fn, 'wb') )

    #pickle.dump(results, open(save_fn, 'wb') ) -> saving after each analysis instead
    print('Analysis results saved in ', save_fn)


def calculate_svms(h, syn_x, syn_u, trial_info, trial_time, num_reps = 20, \
    decode_test = False, decode_rule = False, decode_sample_vs_test = False, analysis = False, test_mode_pulse=False, pulse=0, test_mode_delay=False, stim_num=0):

    """
    Calculates neuronal and synaptic decoding accuracies uisng support vector machines
    sample is the index of the sample motion direction for each trial_length
    rule is the rule index for each trial_length
    """

    lin_clf = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr', shrinking=False, tol=1e-4)

    num_time_steps = len(trial_time)
    decoding_results = {}

    """
    The synaptic efficacy is the product of syn_x and syn_u, will decode sample
    direction from this value
    """
    syn_efficacy = syn_x*syn_u
    combined = np.concatenate((h, syn_efficacy), axis=0)

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



def svm_wraper(lin_clf, h, syn_eff, combo, stim, rule, num_reps, trial_time, analysis=False, test_mode_pulse=False, pulse=0, test_mode_delay=False,stim_num=0):

    """
    Wraper function used to decode sample/test or rule information
    from hidden activity (h) and synaptic efficacies (syn_eff)
    """
    train_pct = 0.75
    trials_per_cond = 25
    _, num_time_steps, num_trials = h.shape
    num_rules = len(np.unique(rule))
    if par['trial_type']=='chunking':
        if analysis:
            num_stim = 1
        elif test_mode_pulse:
            num_stim = pulse
        else:
            num_stim = par['num_pulses']
    else:
        num_stim = par['num_receptive_fields']

    #num_stim = par['num_pulses'] if par['trial_type']=='chunking' else par['num_receptive_fields']

    score_h = np.zeros((num_rules, num_stim, num_reps, num_time_steps), dtype = np.float32)
    score_syn_eff = np.zeros((num_rules, num_stim, num_reps, num_time_steps), dtype = np.float32)
    score_combo = np.zeros((num_rules, num_stim, num_reps, num_time_steps), dtype = np.float32)

    for r in range(num_rules):
        ind_rule = np.where(rule==r)[0]

        for n in range(num_stim):
            if par['trial_type'] == 'dualDMS' or par['trial_type'] == 'chunking':
                if analysis:
                    current_stim = stim[:,stim_num]
                else:
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

                if network_weights['w_out'][n1,n2] <= 0:
                    continue

                # create new dict of weights
                weights_new = {}
                for k,v in network_weights.items():
                    weights_new[k] = np.array(v+1e-32)

                # lesion weights
                q = np.ones((3,par['n_hidden']), dtype=np.float32)
                q[n1,n2] = 0
                weights_new['w_out'] *= q

                # simulate network
                y_hat, _, _, _ = run_model(x_test, hidden_init_test, syn_x_init_test, syn_u_init_test, weights_new)
                lesion_results['lesion_accuracy_out'][r,n1,n2],_,_ = get_perf(y_test, y_hat, train_mask_test)

        print('Lesioning recurrent weights...')
        for n1 in range(par['n_hidden']):
            for n2 in range(par['n_hidden']):

                if network_weights['w_rnn'][n1,n2] <= 0:
                    continue

                weights_new = {}
                for k,v in network_weights.items():
                    weights_new[k] = np.array(v+1e-32)

                # lesion weights
                q = np.ones((par['n_hidden'],par['n_hidden']), dtype=np.float32)
                q[n1,n2] = 0
                weights_new['w_rnn'] *= q

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


def simulate_network(trial_info, h, syn_x, syn_u, network_weights, num_reps = 5):
    """
    Simulation will start from the start of the test period until the end of trial
    """
    onset = [np.unique(np.array(trial_info['timeline']))[-2*p-2] for p in range(par['num_pulses'])][::-1]

    simulation_results = {
        'accuracy'                      : np.zeros((par['num_pulses'], par['n_hidden'], num_reps)),
        'accuracy_neural_shuffled'      : np.zeros((par['num_pulses'], par['n_hidden'], num_reps)),
        'accuracy_syn_shuffled'         : np.zeros((par['num_pulses'], par['n_hidden'], num_reps))}


    for p in range(par['num_pulses']):
        test_onset = onset[p]

        _, trial_length, batch_train_size = h.shape

        train_mask = np.zeros((trial_length, par['batch_train_size']),dtype=np.float32)
        train_mask[onset[p]+par['mask_duration']//par['dt']:onset[p]+par['sample_time']//par['dt']] = 1
        #print(np.sum(train_mask))

        #test_length = trial_length - test_onset
        test_length = par['resp_cue_time']//par['dt']
        trial_ind = np.arange(par['batch_train_size'])

        print('h', h.shape)
        print('trial_length',trial_length)
        print('test_length',test_length)
        print('test_onset',test_onset)
        print('trial_info', trial_info['neural_input'].shape)
        x = np.split(trial_info['neural_input'][:,test_onset:test_onset+test_length,trial_ind],test_length,axis=1)
        y = trial_info['desired_output'][:,test_onset:test_onset+test_length,trial_ind]
        train_mask = train_mask[test_onset:test_onset+test_length]
        #print(np.sum(train_mask))

        for n in range(num_reps):
            print(n, "out of ", num_reps)

            """
            Calculating behavioral accuracy without shuffling
            """
            hidden_init = h[:,test_onset-1,trial_ind]
            syn_x_init = syn_x[:,test_onset-1,trial_ind]
            syn_u_init = syn_u[:,test_onset-1,trial_ind]
            y_hat, _, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
            #print(np.sum(train_mask))
            simulation_results['accuracy'][p,:,n] = get_perf(y, y_hat, train_mask)

            for m in range(par['n_hidden']):
                """
                Keep the synaptic values fixed, permute the neural activity
                """
                ind_shuffle = np.random.permutation(len(trial_ind))
                hidden_init = h[:,test_onset-1,trial_ind]
                hidden_init[m,:] = hidden_init[m, ind_shuffle]
                y_hat, _, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
                simulation_results['accuracy_neural_shuffled'][p,m,n] = get_perf(y, y_hat, train_mask)

                """
                Keep the hidden values fixed, permute synaptic values
                """
                hidden_init = h[:,test_onset-1,trial_ind]
                syn_x_init = syn_x[:,test_onset-1,trial_ind]
                syn_x_init[m,:] = syn_x_init[m,ind_shuffle]
                syn_u_init = syn_u[:,test_onset-1,trial_ind]
                syn_u_init[m,:] = syn_u_init[m,ind_shuffle]
                y_hat, _, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
                simulation_results['accuracy_syn_shuffled'][p,m,n] = get_perf(y, y_hat, train_mask)

    return simulation_results

def cut_weights(x_dict, trial_info, start_time, trial_time, h, syn_x, syn_u, network_weights, num_reps = 1, num_top_neurons = 4):
    """
    Simulation will start from the start of the test period until the end of trial
    """
    onset = np.array([np.unique(np.array(trial_info['timeline']))[-2*p-2] for p in range(par['num_pulses'])][::-1])

    _, trial_length, batch_train_size = h.shape

    eolongd = (par['dead_time']+par['fix_time'] + par['num_pulses'] * par['sample_time'] + (par['num_pulses']-1)*par['delay_time'] + par['long_delay_time'])//par['dt']
    #start_time = eolongd-(par['long_delay_time']//par['dt'])+1
    onset -= start_time

    cutting_results = {
        'cut_neurons'             : np.zeros((par['num_pulses'], num_top_neurons),dtype=np.float32),
        'accuracy_before_cut'     : np.zeros((par['num_pulses'], par['num_pulses'], num_reps),dtype=np.float32),
        'accuracy_after_cut'      : np.zeros((par['num_pulses'], par['num_pulses'], num_reps),dtype=np.float32),
        'synaptic_pev_after_cut'          : np.zeros((par['n_hidden'], par['num_pulses'], trial_length, num_reps),dtype=np.float32),
        'neuronal_pev_after_cut'          : np.zeros((par['n_hidden'], par['num_pulses'], trial_length, num_reps),dtype=np.float32),
        'neuronal_pref_dir_after_cut'     : np.zeros((par['n_hidden'],  par['num_pulses'], trial_length, num_reps), dtype=np.float32),
        'synaptic_pref_dir_after_cut'     : np.zeros((par['n_hidden'],  par['num_pulses'], trial_length, num_reps), dtype=np.float32)}

    h = h[:,start_time,:]
    syn_x = syn_x[:,start_time,:]
    syn_u = syn_u[:,start_time,:]

    x = np.split(trial_info['neural_input'],trial_length,axis=1)
    x = x[start_time:]

    y = trial_info['desired_output']
    y = y[:,start_time:,:]

    train_mask = np.zeros((trial_length-start_time, par['batch_train_size']),dtype=np.float32)
    y_hat, _, _, _ = run_model(x, h, syn_x, syn_u, network_weights)

    for p in range(par['num_pulses']):
        print(p, "out of ", par['num_pulses'], " pulses")

        """
        Calculating behavioral accuracy without shuffling
        """
        train_mask[:,:] = 0
        train_mask[onset[p]+par['mask_duration']//par['dt']:onset[p]+par['sample_time']//par['dt']] = 1
        cutting_results['accuracy_before_cut'][:,p,:] = get_perf(y, y_hat, train_mask)

        """
        Cutting top neurons from synaptic_pev result
        """
        arr = x_dict['synaptic_pev'][:,p,onset[p]+start_time-1]
        #top_ind = arr.argsort()[-num_top_neurons:][::-1]
        top_ind = np.random.choice(100, size=4)
        cutting_results['cut_neurons'][p,:] = top_ind
        print(top_ind)

        cut_weights = copy.deepcopy(network_weights)
        for ind in top_ind:
            cut_weights['w_rnn'][ind,top_ind] = 0
        # print("\n\ndebugging...............")
        # print(cut_weights['w_rnn'][top_ind[0],top_ind[1]])
        # print(cut_weights['w_rnn'][top_ind[0],top_ind[1]] == network_weights['w_rnn'][top_ind[0],top_ind[1]])
        y_hat_cut, h_cut, syn_x_cut, syn_u_cut = run_model(x, h, syn_x, syn_u, cut_weights)

        """
        Calculating behavioral accuracy for each pulse after cut
        """
        for p2 in range(par['num_pulses']):
            print(p2, "out of ", par['num_pulses'])
            train_mask[:,:] = 0
            train_mask[onset[p2]+par['mask_duration']//par['dt']:onset[p2]+par['sample_time']//par['dt']] = 1

            # for n in range(num_reps):
            cutting_results['accuracy_after_cut'][p,p2,:] = get_perf(y, y_hat_cut, train_mask)

        # for n in range(num_reps):
        #tuning_results = calculate_tuning(h_cut, syn_x_cut, syn_u_cut, trial_info, trial_time[start_time:], cut_weights)
        #for key, val in tuning_results.items():
        #    cutting_results[key+"_after_cut"][:,:,:,:] = val


    return cutting_results

def calculate_tuning(h, syn_x, syn_u, trial_info, trial_time, network_weights):

    epsilon = 1e-9
    """
    Calculates neuronal and synaptic sample motion direction tuning
    """

    rule = np.array(trial_info['rule'])
    sample = np.reshape(np.array(trial_info['sample']),(par['batch_train_size'], par['num_pulses']))

    num_time_steps = len(trial_time)

    # want zeros(n_hidden, n_pulse, n_time)

    tuning_results = {
        'neuronal_pref_dir'     : np.zeros((par['n_hidden'],  par['num_pulses'], num_time_steps), dtype=np.float32),
        'synaptic_pref_dir'     : np.zeros((par['n_hidden'],  par['num_pulses'], num_time_steps), dtype=np.float32),
        'neuronal_pev'          : np.zeros((par['n_hidden'],  par['num_pulses'], num_time_steps), dtype=np.float32),
        'synaptic_pev'          : np.zeros((par['n_hidden'],  par['num_pulses'], num_time_steps), dtype=np.float32)}

    mask = np.array(trial_info['train_mask'])

    """
    The synaptic efficacy is the product of syn_x and syn_u, will decode sample
    direction from this value
    """
    syn_efficacy = syn_x*syn_u

    sample_dir = np.ones((par['batch_train_size'], 3, par['num_pulses']))


    for i in range(par['num_pulses']):
        sample_dir[:,1, i] = np.cos(2*np.pi*sample[:,i]/par['num_motion_dirs'])
        sample_dir[:,2, i] = np.sin(2*np.pi*sample[:,i]/par['num_motion_dirs'])


    for n in range(par['n_hidden']):
        for t in range(num_time_steps):
            for i in range(par['num_pulses']):

                # Neuronal sample tuning
                w = np.linalg.lstsq(sample_dir[:,:,i], h[n,t,:])
                w = np.reshape(w[0],(3,1))
                h_hat =  np.dot(sample_dir[:,:,i], w).T
                pred_err = h[n,t,:] - h_hat
                mse = np.mean(pred_err**2) # var (h-h_hat)
                response_var = np.var(h[n,t,:]) # var(h)

                if response_var > epsilon:
                    tuning_results['neuronal_pev'][n,i,t] = 1 - mse/(response_var + epsilon)
                    tuning_results['neuronal_pref_dir'][n,i,t] = np.arctan2(w[2,0],w[1,0])

                # Synaptic sample tuning
                w = np.linalg.lstsq(sample_dir[:,:,i], syn_efficacy[n,t,:])
                w = np.reshape(w[0],(3,1))
                syn_hat = np.dot(sample_dir[:,:,i], w).T
                pred_err = syn_efficacy[n,t,:] - syn_hat
                mse = np.mean(pred_err**2)
                response_var = np.var(syn_efficacy[n,t,:])

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


def get_perf(y, y_hat, mask):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when y[0,:,:] is not 0
    y is the desired output
    y_hat is the actual output
    """
    #print("Entering get_perf...")
    #print(np.sum(mask))
    y_hat = np.stack(y_hat, axis=1)
    mask *= y[0,:,:]==0
    #print(np.sum(mask))
    y = np.argmax(y, axis = 0)
    y_hat = np.argmax(y_hat, axis = 0)
    # print(np.float32(y == y_hat))
    # print(np.sum(np.float32(y == y_hat)*np.squeeze(mask)))
    # print(np.sum(mask))
    accuracy = np.sum(np.float32(y == y_hat)*np.squeeze(mask))/np.sum(mask)

    return accuracy
