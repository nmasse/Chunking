import pickle
import numpy as np
import matplotlib.pyplot as plt
import stimulus


file = 'analysis_restart_no_var_delay_6_acc90.pkl'

x = pickle.load(open("./savedir_restart/"+file, 'rb'))

count = np.zeros((x['parameters']['num_pulses'],x['parameters']['num_pulses']), dtype=np.int8)

par = x['parameters']
end_of_long = (par['dead_time']+par['fix_time'] + par['num_pulses'] * par['sample_time'] + (par['num_pulses']-1)*par['delay_time'] + par['long_delay_time'])//par['dt']
print(end_of_long)

stim = stimulus.Stimulus()
trial_info = stim.generate_trial(analysis = False,num_fixed=0,var_delay=False,var_resp_delay=False,var_num_pulses=False)
onset = np.array([np.unique(np.array(trial_info['timeline']))[-2*p-2] for p in range(par['num_pulses'])][::-1])

threshold_list = [0.25, 0.35,0.45,0.55]

for threshold in threshold_list:
	for i in range(x['parameters']['num_pulses']):
		for j in range(x['parameters']['num_pulses']):
			c_i = np.int8(x['synaptic_pev'][:,i,onset[i]] >= threshold)
			c_j = np.int8(x['synaptic_pev'][:,j,onset[j]] >= threshold)
			count[i,j] = np.sum(c_i*c_j)

	plt.figure()
	plt.imshow(count)
	plt.colorbar()
	plt.title('PEV threshold: {}'.format(threshold))
	plt.savefig('./pev_count/threshold_{}.png'.format(threshold))