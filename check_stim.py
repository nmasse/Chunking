import stimulus
from parameters import *

stim = stimulus.Stimulus()
par['trial_type'] = 'sequence_cue'
par['batch_size'] = 20
par['check_stim'] = True
trial_info = stim.generate_trial(task=par['trial_type'],var_delay=par['var_delay'],var_num_pulses=par['var_num_pulses'])
