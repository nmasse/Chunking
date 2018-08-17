import numpy as np
import matplotlib.pyplot as plt
from parameters import *


class Stimulus:

    def __init__(self):

        # generate tuning functions
        self.motion_tuning, self.fix_tuning, self.order_tuning = self.create_tuning_functions()


    def generate_trial(self, analysis = False, var_delay=False, var_resp_delay=False, var_num_pulses=False, test_mode = False):

        return self.generate_var_chunking_trial(par['num_pulses'], var_delay, var_resp_delay, var_num_pulses, test_mode)


    def generate_var_chunking_trial(self, num_pulses, analysis, var_delay=False, var_resp_delay=False, var_num_pulses=False, test_mode=False):
        """
        Generate trials to investigate chunking
        """

        trial_info = {'desired_output'  :  np.zeros((par['n_output'], par['num_time_steps'], par['batch_train_size']),dtype=np.float32),
                      'train_mask'      :  np.ones((par['num_time_steps'], par['batch_train_size']),dtype=np.float32),
                      'sample'          :  -np.ones((par['batch_train_size'], par['num_max_pulse']),dtype=np.int32),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['n_input'], par['num_time_steps'], par['batch_train_size'])),
                      'pulse_id'        :  -np.ones((par['num_time_steps'], par['batch_train_size']),dtype=np.int8)}

        start = int((par['dead_time'] + par['fix_time'])//par['dt'])
        pulse_dur = int(par['sample_time']//par['dt'])
        resp_dur = int(par['resp_cue_time']//par['dt'])
        mask_dur = int(par['mask_duration']//par['dt'])
        resp_start = int((par['dead_time'] + par['fix_time'] + num_pulses*par['sample_time'] + par['long_delay_time'] + np.sum(par['delay_times']))//par['dt'])
        delay_times = par['delay_times']//par['dt'] if var_delay else par['delay_time']*np.ones_like(par['delay_times'])//par['dt']

        for t in range(par['batch_train_size']):

            """
            Generate trial paramaters
            """
            num_pulses = np.random.choice(range(1,par['num_pulses']+1)) if (var_num_pulses and not test_mode) else par['num_pulses']

            current_delay_times = np.random.permutation(delay_times) if var_delay else delay_times
            stim_times = [range(start + i*pulse_dur + np.sum(current_delay_times[:i]), start + (i+1)*pulse_dur + np.sum(current_delay_times[:i])) for i in range(num_pulses)]


            resp_times = [range(resp_start + 2*i*pulse_dur,resp_start + (2*i+1)*pulse_dur) for i in range(num_pulses)]
            mask_times = [range(resp_start + 2*i*pulse_dur,resp_start + 2*i*pulse_dur + mask_dur) for i in range(num_pulses)]

            trial_info['train_mask'][:par['dead_time']//par['dt'], t] = 0
            trial_info['desired_output'][0, :, t] = 1

            for i in range(num_pulses):

                # stimulus properties
                trial_info['sample'][t,i] = np.random.randint(par['num_motion_dirs'])
                trial_info['neural_input'][:, stim_times[i], t] += np.reshape(self.motion_tuning[:, trial_info['sample'][t,i]],(-1,1))
                if par['order_cue']:
                    trial_info['neural_input'][:, stim_times[i], t] += np.reshape(self.order_tuning[:, i],(-1,1))

                # response properties
                trial_info['pulse_id'][resp_times[i], t] = i
                trial_info['desired_output'][0, resp_times[i], t] = 0
                trial_info['desired_output'][trial_info['sample'][t,i] + 1, resp_times[i], t] = 1
                trial_info['train_mask'][resp_times[i], t] *= par['response_multiplier']
                trial_info['train_mask'][mask_times[i], t] = 0
                if par['num_fix_tuned'] > 0:
                    trial_info['neural_input'][:, resp_times[i], t] += np.reshape(self.fix_tuning[:, 0],(-1,1))
                if par['order_cue']:
                    trial_info['neural_input'][:, resp_times[i], t] += np.reshape(self.order_tuning[:, i],(-1,1))

            # in case there's left over time (true for var pulse conditions)
            trial_info['train_mask'][np.max(resp_times[-1]):, t] = 0

        if False:
            for i in range(5):
                plt.figure()
                #plt.title("num_pulses: "+str(trial_info['num_pulses'][i])+"\nvar_delay: "+str(list(trial_info['delay'][i,:trial_info['num_pulses'][i]-1])+[trial_info['delay'][i,-1]])+"\nresp_delay: "+str(trial_info['resp_delay'][i,:trial_info['num_pulses'][i]]))
                plt.imshow(trial_info['neural_input'][:,:,i],aspect='auto')
                plt.colorbar()
                plt.show()
                plt.close()
                plt.figure()
                plt.plot(trial_info['train_mask'][:,i])
                #plt.title("num_pulses: "+str(trial_info['num_pulses'][i])+"\nvar_delay: "+str(list(trial_info['delay'][i,:trial_info['num_pulses'][i]-1])+[trial_info['delay'][i,-1]])+"\nresp_delay: "+str(trial_info['resp_delay'][i,:trial_info['num_pulses'][i]]))
                plt.show()
                plt.close()
                plt.figure()
                plt.imshow(trial_info['desired_output'][:,:,i],aspect='auto')
                #plt.title("num_pulses: "+str(trial_info['num_pulses'][i])+"\nvar_delay: "+str(list(trial_info['delay'][i,:trial_info['num_pulses'][i]-1])+[trial_info['delay'][i,-1]])+"\nresp_delay: "+str(trial_info['resp_delay'][i,:trial_info['num_pulses'][i]]))
                plt.colorbar()
                plt.show()
                plt.close()

        return trial_info

    def create_tuning_functions(self):

        motion_tuning = np.zeros((par['n_input'], par['num_receptive_fields'], par['num_motion_dirs']))
        fix_tuning = np.zeros((par['n_input'], 1))
        order_tuning = np.zeros((par['n_input'], par['num_pulses']))

        # generate list of prefered directions
        pref_dirs = np.float32(np.arange(0,360,360/(par['num_motion_tuned'])))

        # generate list of possible stimulus directions
        stim_dirs = np.float32(np.arange(0,360,360/par['num_motion_dirs']))

        for n in range(par['num_motion_tuned']):
            for i in range(len(stim_dirs)):
                d = np.cos((stim_dirs[i] - pref_dirs[n])/180*np.pi)
                motion_tuning[n,i] = par['tuning_height']*np.exp(par['kappa']*d)/np.exp(par['kappa'])

        for n in range(par['num_fix_tuned']):
            fix_tuning[par['num_motion_tuned']+n,0] = par['tuning_height']

        for n in range(par['num_rule_tuned']):
            for i in range(par['num_pulses']):
                if n%par['num_pulses'] == i:
                    order_tuning[par['num_motion_tuned']+par['num_fix_tuned']+n,i] = par['tuning_height']


        return motion_tuning, fix_tuning, order_tuning


    def plot_neural_input(self, trial_info):

        print(trial_info['desired_output'][ :, 0, :].T)
        f = plt.figure(figsize=(8,4))
        ax = f.add_subplot(1, 1, 1)
        t = np.arange(0,400+500+2000,par['dt'])
        t -= 900
        t0,t1,t2,t3 = np.where(t==-500), np.where(t==0),np.where(t==500),np.where(t==1500)
        #im = ax.imshow(trial_info['neural_input'][:,0,:].T, aspect='auto', interpolation='none')
        im = ax.imshow(trial_info['neural_input'][:,:,0], aspect='auto', interpolation='none')
        #plt.imshow(trial_info['desired_output'][:, :, 0], aspect='auto')
        ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
        ax.set_xticklabels([-500,0,500,1500])
        ax.set_yticks([0, 9, 18, 27])
        ax.set_yticklabels([0,90,180,270])
        f.colorbar(im,orientation='vertical')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Motion direction')
        ax.set_xlabel('Time relative to sample onset (ms)')
        ax.set_title('Motion input')
        plt.show()
        plt.savefig('stimulus.pdf', format='pdf')
