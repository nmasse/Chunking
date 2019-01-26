import stimulus
from parameters import *
import matplotlib.pyplot as plt

stim = stimulus.Stimulus()
par['trial_type'] = 'chunking'
par['batch_size'] = 20
par['check_stim'] = False
trial_info = stim.generate_trial(analysis = False,num_fixed=0,var_delay=par['var_delay'],var_resp_delay=par['var_resp_delay'],var_num_pulses=par['var_num_pulses'])
plt.figure()
#plt.plot(trial_info['train_mask'][:,0])
plt.plot(trial_info['pulse_masks'][0,:,0])
plt.plot(trial_info['pulse_masks'][1,:,0])
# plt.plot(trial_info['pulse_masks'][2,:,0])
# plt.plot(trial_info['pulse_masks'][3,:,0])
# plt.plot(trial_info['pulse_masks'][4,:,0])
plt.show()