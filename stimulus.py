import numpy as np
from parameters import *
import matplotlib.pyplot as plt


class Stimulus:

    def __init__(self):

        # Generate tuning functions
        self.create_tuning_functions()


    def generate_trial(self, task, var_delay=False, var_num_pulses=False, all_RF=False, test_mode = False):

        if task == "sequence":
            trial_info = self.generate_sequence_trial(var_delay, var_num_pulses, all_RF)
        elif task == "sequence_cue":
            trial_info = self.generate_sequence_cue_trial(var_delay, var_num_pulses)
        elif task == "RF_detection":
            trial_info = self.generate_RF_detection_trial(var_delay)
        elif task == "RF_cue":
            trial_info = self.generate_RF_cue_trial(var_delay)
        else:
            trial_info = None

        # Add a rule signal if desired
        if len(par['trial_type']) > 1 and par['num_rule_tuned'] > 0:
            trial_info['neural_input'] += self.rule_tuning[par['trial_type'].index(task)]

        """
        for b in range(16):
            print(b)
            fig, ax = plt.subplots(3)
            ax[0].imshow(trial_info['neural_input'][:,b,:].T, aspect='auto', clim=[0,4])
            ax[1].imshow(trial_info['desired_output'][:,b,:].T, aspect='auto', clim=[0,4])
            ax[2].imshow(trial_info['train_mask'][:,b,np.newaxis].T, aspect='auto', clim=[0,4])

            plt.show()
        quit()
        #"""

        return trial_info


    def generate_sequence_trial(self, var_delay=False, var_num_pulses=False, all_RF=False, test_mode=False):
        trial_info = {'desired_output'  :  np.zeros((par['num_time_steps'], par['batch_train_size'], par['n_output']),dtype=np.float32),
                      'train_mask'      :  np.ones((par['num_time_steps'], par['batch_train_size']),dtype=np.float32),
                      'sample'          :  -np.ones((par['batch_train_size'], par['num_pulses']),dtype=np.int32),
                      'sample_RF'       : np.zeros((par['batch_train_size'], par['num_pulses']),dtype=np.int32),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['num_time_steps'], par['batch_train_size'], par['n_input'])),
                      'pulse_id'        :  -np.ones((par['num_time_steps'], par['batch_train_size']),dtype=np.int8),
                      'test'            : np.zeros((par['batch_train_size']),dtype=np.int32)}

        num_pulses = par['num_pulses']
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
            num_pulses = par['num_pulses']

            current_delay_times = np.random.permutation(delay_times) if var_delay else delay_times
            stim_times = [range(start + i*pulse_dur + np.sum(current_delay_times[:i]), start + (i+1)*pulse_dur + np.sum(current_delay_times[:i])) for i in range(num_pulses)]

            resp_times = [range(resp_start + 2*i*pulse_dur,resp_start + (2*i+1)*pulse_dur) for i in range(num_pulses)]
            mask_times = [range(resp_start + 2*i*pulse_dur,resp_start + 2*i*pulse_dur + mask_dur) for i in range(num_pulses)]

            trial_info['train_mask'][:par['dead_time']//par['dt'], t] = 0
            trial_info['neural_input'][:,t,par['num_motion_tuned']*par['num_RFs']:par['num_motion_tuned']*par['num_RFs']+par['num_fix_tuned']] = par['tuning_height'] #self.fix_tuning[:, 0]

            trial_info['desired_output'][:, t, :] = self.fix_output_tuning

            resp_count = 0
            for i in range(num_pulses):

                if not var_num_pulses or np.random.rand() < par['pulse_prob']:

                    # stimulus properties
                    trial_info['sample'][t,i] = np.random.randint(par['num_motion_dirs'])
                    trial_info['sample_RF'][t,i] = 0

                    trial_info['neural_input'][stim_times[i], t, :] += self.motion_tuning[trial_info['sample'][t,i], trial_info['sample_RF'][t,i]]

                    # response properties
                    trial_info['pulse_id'][resp_times[resp_count], t] = i
                    trial_info['desired_output'][resp_times[resp_count], t, :] = self.dir_output_tuning[trial_info['sample'][t,i]]
                    trial_info['neural_input'][resp_times[resp_count], t, :] += self.cue_tuning[:, trial_info['test'][t]]
                    trial_info['train_mask'][resp_times[resp_count], t] *= par['response_multiplier']
                    trial_info['train_mask'][mask_times[resp_count], t] = 0

                    if par['num_fix_tuned'] > 0:
                        trial_info['neural_input'][resp_times[resp_count], t, par['num_motion_tuned']*par['num_RFs']:par['num_motion_tuned']*par['num_RFs']+par['num_fix_tuned']] = 0

                    resp_count += 1

            # in case there's left over time (true for var pulse conditions)
            trial_info['train_mask'][np.max(resp_times[-1])+1:, t] = 0

        if False:
            self.plot_stim(trial_info)

        return trial_info


    def generate_sequence_cue_trial(self, var_delay=False, var_num_pulses=False, all_RF=False, test_mode=False):
        trial_info = {'desired_output'  :  np.zeros((par['num_time_steps'], par['batch_train_size'], par['n_output']),dtype=np.float32),
                      'train_mask'      :  np.ones((par['num_time_steps'], par['batch_train_size']),dtype=np.float32),
                      'sample'          :  -np.ones((par['batch_train_size'], par['num_pulses']),dtype=np.int32),
                      'test'            : np.zeros((par['batch_train_size']),dtype=np.int32),
                      'sample_RF'       : np.zeros((par['batch_train_size'], par['num_pulses']),dtype=np.int32),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['num_time_steps'], par['batch_train_size'], par['n_input'])),
                      'pulse_id'        :  -np.ones((par['num_time_steps'], par['batch_train_size']),dtype=np.int8)}

        num_pulses = par['num_pulses']
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
            num_pulses = par['num_pulses']
            loc = np.random.permutation(np.arange(par['num_pulses'])) if all_RF else np.array([np.random.choice(np.arange(par['num_pulses']))] * par['num_pulses'])


            current_delay_times = np.random.permutation(delay_times) if var_delay else delay_times
            stim_times = [range(start + i*pulse_dur + np.sum(current_delay_times[:i]), start + (i+1)*pulse_dur + np.sum(current_delay_times[:i])) for i in range(num_pulses)]


            resp_times = [range(resp_start,resp_start+resp_dur)]
            mask_times = [range(resp_start,resp_start+mask_dur)]

            trial_info['train_mask'][:par['dead_time']//par['dt'], t] = 0
            trial_info['neural_input'][:,t,par['num_motion_tuned']*par['num_RFs']:par['num_motion_tuned']*par['num_RFs']+par['num_fix_tuned']] = par['tuning_height'] #self.fix_tuning[:, 0]
            trial_info['desired_output'][:,t,:] = self.fix_output_tuning

            pulse_list = []
            for i in range(num_pulses):

                if not var_num_pulses or np.random.rand() < par['pulse_prob']:

                    # stimulus properties
                    trial_info['sample'][t,i] = np.random.randint(par['num_motion_dirs'])
                    trial_info['sample_RF'][t,i] = loc[i]

                    trial_info['neural_input'][stim_times[i], t, :] += self.motion_tuning[trial_info['sample'][t,i], trial_info['sample_RF'][t,i]]

                    pulse_list.append(trial_info['sample'][t,i])

            if pulse_list == []:
                i = np.random.choice(num_pulses)

                # stimulus properties
                trial_info['sample'][t,i] = np.random.randint(par['num_motion_dirs'])
                trial_info['sample_RF'][t,i] = loc[i]

                trial_info['neural_input'][stim_times[i], t, :] += self.motion_tuning[trial_info['sample'][t,i], trial_info['sample_RF'][t,i]]

                pulse_list.append(trial_info['sample'][t,i])


            trial_info['test'][t] = np.random.choice(np.arange(len(pulse_list)))
            trial_info['pulse_id'][resp_times[0],t] = trial_info['test'][t]

            # response properties
            trial_info['neural_input'][resp_times[0], t, :] += self.cue_tuning[:, trial_info['test'][t]]
            trial_info['desired_output'][resp_times[0], t, :] = self.dir_output_tuning[pulse_list[trial_info['test'][t]]]
            print(resp_times)

            #trial_info['desired_output'][0, resp_times[0], t] = 0
            #trial_info['desired_output'][trial_info['sample'][t,trial_info['test'][t]] + 1, resp_times[0], t] = self.dir_output_tuning[trial_info['sample'][t,trial_info['test'][t]]]
            trial_info['train_mask'][resp_times[0], t] *= par['response_multiplier']
            trial_info['train_mask'][mask_times[0], t] = 0

            if par['num_fix_tuned'] > 0:
                trial_info['neural_input'][resp_times[0], t, par['num_motion_tuned']*par['num_RFs']:par['num_motion_tuned']*par['num_RFs']+par['num_fix_tuned']] = 0

            # in case there's left over time (true for var pulse conditions)
            trial_info['train_mask'][np.max(resp_times[-1])+1:, t] = 0

        if False:
            self.plot_stim(trial_info)

        return trial_info


    def generate_RF_detection_trial(self, var_delay=True):

        trial_info = {'desired_output'  :  np.zeros((par['num_time_steps'], par['batch_train_size'], par['n_output']),dtype=np.float32),
                      'train_mask'      :  np.ones((par['num_time_steps'], par['batch_train_size']),dtype=np.float32),
                      'sample'          :  -np.ones((par['batch_train_size'], par['num_pulses']),dtype=np.int32),
                      'sample_RF'       : np.zeros((par['batch_train_size'], par['num_pulses']),dtype=np.int32),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['num_time_steps'], par['batch_train_size'], par['n_input'])),
                      'pulse_id'        :  -np.ones((par['num_time_steps'], par['batch_train_size']),dtype=np.int8),
                      'test'            : np.zeros((par['batch_train_size']),dtype=np.int32)}

        dead            = int(par['dead_time']//par['dt'])
        start           = int((par['dead_time'] + par['fix_time'])//par['dt'])
        pulse_dur       = int(par['sample_time_RF']//par['dt'])
        delay_dur       = int(par['RF_long_delay_time']//par['dt'])
        var_delay_max   = int(par['var_delay_max']//par['dt'])
        mask_dur        = int(par['mask_duration']//par['dt'])
        resp_start      = start + pulse_dur + delay_dur

        end_of_var_delay = start + pulse_dur + delay_dur + var_delay_max
        total_length = start + pulse_dur + delay_dur + var_delay_max + pulse_dur


        directions = np.random.choice(par['num_motion_dirs'], size=[par['batch_train_size'], par['num_RFs']])
        targets = np.random.choice(par['num_RFs'], size=[par['batch_train_size']])

        trial_info['sample'] = directions   # Direction in each RF
        trial_info['test']   = targets      # RF to be changed


        trial_info['train_mask'][:dead,:] = 0           # Dead time
        trial_info['train_mask'][total_length:,:] = 0   # Post-task time
        for b in range(par['batch_train_size']):

            # Select response onset
            catch = False
            if var_delay:
                s = np.int32(np.random.exponential(scale=par['var_delay_scale']))
                trial_resp_start = resp_start + s
                if s >= var_delay_max:
                    catch = True    # If catch, mask from end of var_delay_max onward
                    trial_resp_start = -1
            else:
                trial_resp_start = resp_start

            # Make second direction set
            new_directions = np.copy(directions[b,:])
            options = list(set(range(par['num_motion_dirs']))-set([directions[b,targets[b]]]))
            new_directions[targets[b]] = np.random.choice(options)

            trial_info['sample_RF'][b,:] = new_directions   # Changed direction in context

            # Stimulus and response
            stim1 = np.sum([self.motion_tuning[directions[b,rf],rf]/par['num_RFs'] for rf in range(par['num_RFs'])], axis=0)[np.newaxis,:]
            stim2 = np.sum([self.motion_tuning[new_directions[rf],rf]/par['num_RFs'] for rf in range(par['num_RFs'])], axis=0)[np.newaxis,:]
            resp = self.rf_output_tuning[targets[b]]

            # Building neural input
            trial_info['neural_input'][start:start+pulse_dur,b,:] += stim1
            trial_info['neural_input'][trial_resp_start:,b,:] += stim2
            trial_info['neural_input'][:trial_resp_start,b,:] += self.fix_tuning

            # Building network output
            trial_info['desired_output'][:trial_resp_start,b,:] = self.fix_output_tuning
            trial_info['desired_output'][trial_resp_start:,b,:] = resp
            trial_info['pulse_id'][trial_resp_start:,b] = targets[b]

            # Building network mask
            if catch:
                trial_info['train_mask'][end_of_var_delay:,b] = 0.
            else:
                trial_info['train_mask'][trial_resp_start+mask_dur:,b] *= par['response_multiplier']
                trial_info['train_mask'][trial_resp_start+mask_dur+pulse_dur:,b] = 0.
            trial_info['train_mask'][trial_resp_start:trial_resp_start+mask_dur,b] = 0
            trial_info['train_mask'][-1,b] = 0

        return trial_info


    def generate_RF_cue_trial(self, var_delay=True):

        trial_info = {'desired_output'  :  np.zeros((par['num_time_steps'], par['batch_train_size'], par['n_output']),dtype=np.float32),
                      'train_mask'      :  np.ones((par['num_time_steps'], par['batch_train_size']),dtype=np.float32),
                      'sample'          :  -np.ones((par['batch_train_size'], par['num_pulses']),dtype=np.int32),
                      'sample_RF'       : np.zeros((par['batch_train_size'], par['num_pulses']),dtype=np.int32),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['num_time_steps'], par['batch_train_size'], par['n_input'])),
                      'pulse_id'        :  -np.ones((par['num_time_steps'], par['batch_train_size']),dtype=np.int8),
                      'test'            : np.zeros((par['batch_train_size']),dtype=np.int32)}

        dead            = int(par['dead_time']//par['dt'])
        start           = int((par['dead_time'] + par['fix_time'])//par['dt'])
        pulse_dur       = int(par['sample_time_RF']//par['dt'])
        delay_dur       = int(par['RF_long_delay_time']//par['dt'])
        var_delay_max   = int(par['var_delay_max']//par['dt'])
        mask_dur        = int(par['mask_duration']//par['dt'])
        resp_start      = start + pulse_dur + delay_dur

        end_of_var_delay = start + pulse_dur + delay_dur + var_delay_max
        total_length = start + pulse_dur + delay_dur + var_delay_max + pulse_dur

        directions = np.random.choice(par['num_motion_dirs'], size=[par['batch_train_size'], par['num_RFs']])
        targets = np.random.choice(par['num_RFs'], size=[par['batch_train_size']])

        trial_info['sample'] = directions   # Direction in each RF
        trial_info['test']   = targets      # RF to be cued/recalled

        trial_info['train_mask'][:dead,:] = 0           # Dead time
        trial_info['train_mask'][total_length:,:] = 0   # Post-task time
        for b in range(par['batch_train_size']):

            # Select response onset
            catch = False
            if var_delay:
                s = np.int32(np.random.exponential(scale=par['var_delay_scale']))
                trial_resp_start = resp_start + s
                if s >= var_delay_max:
                    catch = True    # If catch, mask from end of var_delay_max onward
                    trial_resp_start = -1
            else:
                trial_resp_start = resp_start

            # Stimulus and response
            stim = np.sum([self.motion_tuning[directions[b,rf],rf]/par['num_RFs'] for rf in range(par['num_RFs'])], axis=0)[np.newaxis,:]
            resp = self.dir_output_tuning[directions[b,targets[b]]]

            # Building neural input
            trial_info['neural_input'][start:start+pulse_dur,b,:] += stim
            trial_info['neural_input'][trial_resp_start:,b,:] += self.cue_tuning[:, targets[b]]
            trial_info['neural_input'][:trial_resp_start,b,:] += self.fix_tuning

            # Building network output
            trial_info['desired_output'][:trial_resp_start,b,:] = self.fix_output_tuning
            trial_info['desired_output'][trial_resp_start:,b,:] = resp
            trial_info['pulse_id'][trial_resp_start:,b] = targets[b]

            # Building network mask
            if catch:
                trial_info['train_mask'][end_of_var_delay:,b] = 0.
            else:
                trial_info['train_mask'][trial_resp_start+mask_dur:,b] *= par['response_multiplier']
                trial_info['train_mask'][trial_resp_start+mask_dur+pulse_dur:,b] = 0.
            trial_info['train_mask'][trial_resp_start:trial_resp_start+mask_dur,b] = 0
            trial_info['train_mask'][-1,b] = 0

        return trial_info


    def create_tuning_functions(self):

        # Motion tuning     --> directional preferences
        # Fixation tuning   --> just fixation
        # Rule tuning       --> current task
        # Dir output tuning --> sin/cos directional outputs based on motion
        # RF output tuning  --> sin/cos directional outputs based on receptive field
        # Order tuning      --> order cue for sequence task

        motion_tuning     = np.zeros([par['num_motion_dirs'], par['num_RFs'], par['n_input']])
        fix_tuning        = np.zeros([1,par['n_input']])
        rule_tuning       = np.zeros([par['num_rule_tuned'],par['n_input']])
        fix_output_tuning = np.zeros([1,par['n_output']])
        dir_output_tuning = np.zeros([par['num_motion_dirs'], par['n_output']])
        rf_output_tuning  = np.zeros([par['num_RFs'], par['n_output']])
        cue_tuning        = np.zeros([par['n_input'], par['num_pulses']])

        # Generate lists of preferred and possible stimulus directions
        pref_dirs = np.float32(np.arange(0,2*np.pi,2*np.pi/par['num_motion_tuned']))
        stim_dirs = np.float32(np.arange(0,2*np.pi,2*np.pi/par['num_motion_dirs']))
        rf_dirs   = np.float32(np.arange(0,2*np.pi,2*np.pi/par['num_RFs']))

        # Tune individual neurons to specific stimulus directions
        for n in range(par['num_motion_tuned']):
            diff = np.cos(stim_dirs-pref_dirs[n])
            for r in range(par['num_RFs']):
                motion_tuning[:,r,r*par['num_motion_tuned']+n] = par['tuning_height']*np.exp(par['kappa']*diff)/np.exp(par['kappa'])

        # Tune fixation neurons to the correct height
        # for n in range(par['num_fix_tuned']):
        fix_tuning[0,par['total_motion_tuned']] = par['tuning_height']

        if par['output_type'] == 'directional':
            fix_output_tuning[0,:] = 0.
        elif par['output_type'] == 'one_hot':
            fix_output_tuning[0,0] = 1.

        # Tune rule neurons to the correct height
        for n in range(par['num_rule_tuned']):
            rule_tuning[n,par['total_motion_tuned']+par['num_fix_tuned']+par['num_cue_tuned']+n] = par['tuning_height']

        # Tune output neurons to the correct directions
        for d in range(par['num_motion_dirs']):
            if par['output_type'] == 'directional':
                dir_output_tuning[d] = [np.cos(stim_dirs[d]), np.sin(stim_dirs[d])]
            elif par['output_type'] == 'one_hot':
                dir_output_tuning[d, d+1] = 1.

        # Tune output neurons to the correct directions
        # for rf in range(par['num_RFs']):
        #     if par['output_type'] == 'directional':
        #         rf_output_tuning[rf] = [np.cos(rf_dirs[rf]), np.sin(rf_dirs[rf])]
        #     elif par['output_type'] == 'one_hot':
        #         rf_output_tuning[rf, rf+par['num_motion_dirs']+1] = 1.

        # Tune order neurons
        # for n in range(par['num_cue_tuned']):
        for i in range(par['num_pulses']):
                # if n%par['num_pulses'] == i:
            cue_tuning[par['num_motion_tuned']*par['num_RFs']+par['num_fix_tuned'],i] = par['tuning_height']

        # Set tunings to class elements
        self.motion_tuning     = motion_tuning
        self.fix_tuning        = fix_tuning
        self.rule_tuning       = rule_tuning
        self.dir_output_tuning = dir_output_tuning
        # self.rf_output_tuning  = rf_output_tuning
        self.fix_output_tuning = fix_output_tuning
        self.cue_tuning        = cue_tuning


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

    def plot_stim(self, trial_info):
        for i in range(5):
            fig, ax = plt.subplots(3)

            ax[0].imshow(trial_info['neural_input'][:,i,:],aspect='auto', clim=[0,par['tuning_height']])
            ax[1].imshow(trial_info['train_mask'][:,i,np.newaxis],aspect='auto', clim=[0,par['tuning_height']])
            ax[2].imshow(trial_info['desired_output'][:,i,:],aspect='auto', clim=[0,1])

            ax[0].set_title('Neural Input')
            ax[1].set_title('Train Mask')
            ax[2].set_title('Desired Output')

            plt.show()


if __name__ == '__main__':
    s = Stimulus()
    trial_info = s.generate_trial('sequence', var_delay=False, var_num_pulses=False)
    s.plot_stim(trial_info)
