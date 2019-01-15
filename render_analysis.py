import numpy as np
import pickle
import os, sys
from parameters import *
import matplotlib.pyplot as plt
from itertools import product
import stimulus

### Script to run is at the bottom

def task_overlap(threshold=0.25, all=False):

    def plot_now(save_fn):

        task_text = task if not 'all' in save_fn else 'all tasks'

        pulse_range = par['num_pulses'] if not all else par['num_pulses']*len(par['trial_type'])
        o = np.zeros([pulse_range, pulse_range])
        for i, j in product(range(pulse_range), range(pulse_range)):
            o[i,j] = len(np.intersect1d(z[i],z[j]))

        plt.figure(figsize=(8,6))
        plt.imshow(o, aspect='auto')
        plt.colorbar()
        plt.suptitle('{} PEV overlap for {} (threshold={})'.format(pev_type, task_text, threshold))
        if 'all' in save_fn:
            plt.title('({})'.format(', '.join(par['trial_type'])), fontsize=10)
        plt.xticks(np.arange(pulse_range))
        plt.yticks(np.arange(pulse_range))
        plt.savefig(save_fn)
        plt.clf()
        plt.close()


    for pev_type in ['neuronal', 'synaptic']:

        z = []
        for task in par['trial_type']:

            pev = data[task]['{}_pev'.format(pev_type)]

            if pev_type == 'synaptic':
                end_of_delay = np.where(trial_info[task]['desired_output'][:,:,0]==0.)[0][0]
                if all:
                    z += [np.where(pev[:,i,end_of_delay]>threshold)[0] for i in range(par['num_pulses'])]
                else:
                    z = [np.where(pev[:,i,end_of_delay]>threshold)[0] for i in range(par['num_pulses'])]

            elif pev_type == 'neuronal':

                if 'sequence' in task:
                    if not all:
                        z = []
                    pulse_start = int((par['dead_time'] + par['fix_time'])//par['dt'])
                    sample_time = int(par['sample_time']//par['dt'])
                    delay_time  = int(par['delay_time']//par['dt'])
                    for i in range(par['num_pulses']):
                        t0 = pulse_start + i*sample_time + np.maximum(0,(i-1))*delay_time
                        t1 = pulse_start + (i+1)*sample_time + i*delay_time
                        z.append(np.where(np.mean(pev[:,i,t0:t1],axis=-1)>threshold)[0])

                elif 'RF' in task:
                    pulse_start = int((par['dead_time'] + par['fix_time'])//par['dt'])
                    sample_time = int(par['sample_time_RF']//par['dt'])
                    t0 = pulse_start
                    t1 = pulse_start + sample_time
                    if all:
                        z += [np.where(np.mean(pev[:,i,t0:t1],axis=-1)>threshold)[0] for i in range(par['num_pulses'])]
                    else:
                        z = [np.where(np.mean(pev[:,i,t0:t1],axis=-1)>threshold)[0] for i in range(par['num_pulses'])]

            if not all:
                plot_now('./plots/{}/{}_{}_pev_overlap_threshold{}.png'.format(foldername, task, pev_type, 100*threshold))

        if all:
            plot_now('./plots/{}/all_tasks_{}_pev_overlap_threshold{}.png'.format(foldername, pev_type, 100*threshold))


def task_pevs():

    for task in par['trial_type']:
        x = data[task]
        if 'RF' in task:
            aspect = 'RF'
        elif 'sequence' in task:
            aspect = 'Pulse'

        for pev_type in ['synaptic', 'neuronal']:

            key = '{}_pev'.format(pev_type)

            fig, ax = plt.subplots(2,par['num_pulses']//2+par['num_pulses']%2,figsize=(8,6))
            hori = par['num_pulses']//2+par['num_pulses']%2
            for p in range(par['num_pulses']):

                if par['num_pulses'] == 4:
                    ax[p//2,p%hori].imshow(x[key][:,p,:], aspect='auto', clim=[0,1])
                    ax[p//2,p%hori].set_title('{} {}'.format(aspect, p))
                elif par['num_pulses'] == 5:
                    ax[0 if p <= par['num_pulses']//2 else 1,p%hori].imshow(x[key][:,p,:], aspect='auto', clim=[0,1])
                    ax[0 if p <= par['num_pulses']//2 else 1,p%hori].set_title('{} {}'.format(aspect, p))

            plt.suptitle('{} PEV for {}'.format(pev_type, task))
            plt.savefig('./plots/{}/{}_{}_pev.png'.format(foldername, task, pev_type))
            plt.clf()
            plt.close()


def task_currents():

    rnn_currents = ['rnn_current', 'exc_current', 'inh_current']
    inp_currents = ['motion_current', 'fix_current', 'cue_current']
    for task in par['trial_type']:
        x = data[task]

        fig, ax = plt.subplots(2,1,figsize=(8,7))

        for c, k in zip(['r', 'g', 'b'], rnn_currents):
            current = x[k]

            time = np.arange(current.shape[0])
            curve0 = np.mean(current[:,:,0], axis=1)
            curve1 = np.mean(current[:,:,1], axis=1)
            ax[0].plot(time, curve0, c=c, label=k)
            ax[1].plot(time, curve1, c=c, label=k)

        for c, k in zip(['m', 'y', 'c'], inp_currents):
            current = x[k]

            time = np.arange(current.shape[0])
            curve = np.mean(current, axis=1)
            ax[0].plot(time, curve, c=c, label=k)
            ax[1].plot(time, curve, c=c, label=k)

        ax[0].set_title('RNN Currents')
        ax[1].set_title('Effective RNN Currents')
        ax[0].legend(loc='upper right', ncol=2)
        ax[1].legend(loc='upper right', ncol=2)
        plt.suptitle('Network Currents for {} task'.format(task))
        plt.savefig('./plots/{}/{}_currents.png'.format(foldername, task))
        plt.clf()
        plt.close()


def selective_task_currents(num_top_neurons=5):

    rnn_currents = ['rnn_current', 'exc_current', 'inh_current']
    inp_currents = ['motion_current', 'fix_current', 'cue_current']

    for task in par['trial_type']:
        x = data[task]
        pev = x['synaptic_pev']
        end_of_task = np.where(trial_info[task]['train_mask'][:,0]==1.)[0][-1]

        for p in range(par['num_pulses']):

            mean_pev = np.mean(pev[:,p,:end_of_task+1], axis=-1)
            greatest_neurons = np.argsort(mean_pev)[-(num_top_neurons):][::-1]

            fig, ax = plt.subplots(2,1,figsize=(8,7))

            for c, k in zip(['r', 'g', 'b'], rnn_currents):
                current = x[k]

                time = np.arange(current.shape[0])
                curve0 = np.mean(current[:,greatest_neurons,0], axis=1)
                curve1 = np.mean(current[:,greatest_neurons,1], axis=1)
                ax[0].plot(time, curve0, c=c, label=k)
                ax[1].plot(time, curve1, c=c, label=k)

            for c, k in zip(['m', 'y', 'c'], inp_currents):
                current = x[k]

                time = np.arange(current.shape[0])
                curve = np.mean(current[:,greatest_neurons], axis=1)
                ax[0].plot(time, curve, c=c, label=k)
                ax[1].plot(time, curve, c=c, label=k)

            ax[0].set_title('RNN Currents')
            ax[1].set_title('Effective RNN Currents')
            ax[0].legend(loc='upper right', ncol=2)
            ax[1].legend(loc='upper right', ncol=2)

            aspect = 'RF' if 'RF' in task else 'pulse'
            plt.suptitle('Network Currents for Top {} PEV Neurons for {} task, {} {}'.format(num_top_neurons, task, aspect, p))
            plt.savefig('./plots/{}/top_synaptic_PEV_currents/{}top/{}_pulse{}_{}top_currents.png'.format(foldername, num_top_neurons, task, p, num_top_neurons))
            plt.clf()
            plt.close()


def hidden_state_deviation():

    for task in par['trial_type']:
        x = data[task]
        fig, ax = plt.subplots(1, figsize=(8,6))
        im = ax.imshow(x['std_h'].T, aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.set_title('Standard deviation (across trials) for {}'.format(task))
        plt.savefig('./plots/{}/{}_variance.png'.format(foldername, task))
        plt.clf()
        plt.close()


def selective_hidden_state_deviation(num_top_neurons=5):

    for task in par['trial_type']:
        x = data[task]
        pev = x['synaptic_pev']
        end_of_task = np.where(trial_info[task]['train_mask'][:,0]==1.)[0][-1]

        aspect = 'RF' if 'RF' in task else 'pulse'
        fig, ax = plt.subplots(3,2,figsize=(8,7))
        for p in range(par['num_pulses']):

            mean_pev = np.mean(pev[:,p,:end_of_task+1], axis=-1)
            greatest_neurons = np.argsort(mean_pev)[-(num_top_neurons):][::-1]

            im = ax[p//2,p%2].imshow(x['std_h'][:,greatest_neurons].T, aspect='auto')
            ax[p//2,p%2].set_yticks(np.arange(len(greatest_neurons)))
            ax[p//2,p%2].set_yticklabels(greatest_neurons)
            ax[p//2,p%2].set_title('{} {}'.format(aspect, p))
            ax[p//2,p%2].set_ylabel('Top Neurons')

            fig.colorbar(im, ax=ax[p//2,p%2])

        plt.suptitle('Standard Deviation for Top {} PEV Neurons for {} task'.format(num_top_neurons, task))
        plt.savefig('./plots/{}/top_synaptic_PEV_deviations/{}_{}top_deviations.png'.format(foldername, task, num_top_neurons))
        plt.clf()
        plt.close()


def multi_accuracy_selective_hidden_state_deviation():

    pevs = {}
    stds = {}
    for acc in [90, 95, 96, 97, 98]:
        task_pevs = {}
        task_stds = {}
        filename = 'analysis_RF_cue_sequence_cue_p4_100_neuron_high_lr_v0_acc{}.pkl'.format(str(acc))
        data = pickle.load(open('./savedir/new/'+filename, 'rb'))

        for task in par['trial_type']:
            task_pevs[task] = data[task]['synaptic_pev']
            task_stds[task] = data[task]['std_h']

        pevs[acc] = task_pevs
        stds[acc] = task_stds

    for task in par['trial_type'][::-1]:
        end_of_delay = np.where(trial_info[task]['desired_output'][:,:,0]==0.)[0][0] - 1
        fig, ax = plt.subplots(2,2,figsize=(10,7))
        for p in range(par['num_pulses']):

            target = ax[p//2,p%2]

            assembly = []
            for acc in [90, 95, 96, 97, 98]:

                greatest_neurons = np.argsort(pevs[acc][task][:,p,end_of_delay])[-5:][::-1]
                greatest_pevs = pevs[acc][task][greatest_neurons,p,end_of_delay]
                for n, pe in zip(greatest_neurons, greatest_pevs):
                    assembly.append((n, pe))

            aggregate = []
            for a in sorted(assembly, key=lambda x: -x[1]):
                if a[0] not in aggregate:
                    aggregate.append(a[0])
                if len(aggregate) == 5:
                    break

            for acc in [90, 95, 96, 97, 98]:

                aggregate_pevs = pevs[acc][task][aggregate,p,end_of_delay]

                curve = stds[acc][task][:,aggregate]
                #curve = stds[acc][task]*pevs[acc][task][np.newaxis,:,p,end_of_delay]**2/np.sum(pevs[acc][task][:,p,end_of_delay]**2)

                target.plot(np.mean(curve, axis=-1), label='Acc={}'.format(acc))
                target.set_ylabel('Normalized Std. Devs.')

        plt.legend()
        plt.suptitle('Weighted Mean of Std. Dev. by PEV : {}'.format(task))
        plt.show()
















if __name__ == '__main__':

    # Setup
    # filename = 'analysis_var_delay_5_acc90.pkl'
    
    files = os.listdir('./analysis_results/')
    for filename in files:
        print("PLOTTING FILE ", filename)
        foldername = filename[:-4]
        os.makedirs('./plots/{}/'.format(foldername), exist_ok=True)
        os.makedirs('./plots/{}/top_synaptic_PEV_deviations/'.format(foldername), exist_ok=True)
        
        data = pickle.load(open('./analysis_results/' + filename, 'rb'))
        data['parameters']['load_prev_weights'] = False
        update_parameters(data['parameters'])

        trial_info = {}
        stim = stimulus.Stimulus()
        for t in par['trial_type']:
            trial_info[t] = stim.generate_trial(t, var_delay=False, var_num_pulses=False, all_RF=par['all_RF'])

        #multi_accuracy_selective_hidden_state_deviation()
        #quit()

        # Make plots
        task_currents()

        hidden_state_deviation()

        for i in range(6):
            os.makedirs('./plots/{}/top_synaptic_PEV_currents/{}top/'.format(foldername,i+1), exist_ok=True)
            selective_task_currents(num_top_neurons=i+1)

        selective_hidden_state_deviation(num_top_neurons=5)

        for t in [0.15, 0.25]:
            task_overlap(threshold=t)
            task_overlap(threshold=t, all=True)

        task_pevs()
