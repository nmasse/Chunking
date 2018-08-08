import stimulus
from parameters import *

stim = stimulus.Stimulus()
par['trial_type'] = 'chunking'
par['batch_size'] = 20
par['check_stim'] = True
trial_info = stim.generate_trial(analysis = False,num_fixed=0,var_delay=par['var_delay'],var_resp_delay=par['var_resp_delay'],var_num_pulses=par['var_num_pulses'])
