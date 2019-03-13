import pickle
import numpy as np
import matplotlib.pyplot as plt
import stimulus
import matplotlib.cm as cmx
import matplotlib.colors as colors
from parameters import *

bupu = plt.get_cmap('BuPu')
cNorm = colors.Normalize(vmin=0, vmax=range(8)[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=bupu)

color_list = []
for i in range(8):
	colorVal = scalarMap.to_rgba(range(8)[i])
	color_list.append(colorVal)

file = 'analysis_restart_new_var_pulse_5_d300_tc50_acc80.pkl'

x = pickle.load(open("./savedir_restart/"+file, 'rb'))
update_parameters(x['parameters'])

count = np.zeros((x['parameters']['num_pulses'],x['parameters']['num_pulses']), dtype=np.int8)

par = x['parameters']
end_of_delay = (par['dead_time']+par['fix_time'] + par['num_pulses'] * par['sample_time'] + (par['num_pulses']-1)*par['delay_time'] + par['long_delay_time'])//par['dt']
end_of_delay -= 1

stim = stimulus.Stimulus()
trial_info = stim.generate_trial(analysis=False, num_fixed=0, test_mode_pulse=False, test_mode_delay=True)

# onset = np.array([np.unique(np.array(trial_info['timeline']))[-2*p-2] for p in range(par['num_pulses'])][::-1])
# print(onset)
onset = np.array([np.unique(np.array(trial_info['timeline']))[2*p+1] for p in range(par['num_pulses'])]) + 19

threshold_list = [0.2, 0.3, 0.4, 0.5]

for threshold in threshold_list:
	for i in range(x['parameters']['num_pulses']):
		for j in range(x['parameters']['num_pulses']):
			c_i = np.int8(x['neuronal_pev'][:,i,onset[i]] >= threshold)
			c_j = np.int8(x['neuronal_pev'][:,j,onset[i]] >= threshold)
			count[i,j] = np.sum(c_i*c_j)

	plt.figure()
	plt.imshow(count)
	plt.colorbar()
	plt.xticks(np.arange(5), [1,2,3,4,5])
	plt.yticks(np.arange(5), [1,2,3,4,5])
	plt.ylabel("Item")
	plt.xlabel("Item")
	plt.title('Count of Neurons Strongly Encoding for Each Pair of Items')
	plt.savefig('./pev_count/neuronal_{}_acc80.png'.format(threshold))

for threshold in threshold_list:
	for i in range(x['parameters']['num_pulses']):
		for j in range(x['parameters']['num_pulses']):
			# neuron x pulse x time
			c_i = np.int8(x['synaptic_pev'][:,i,onset[i]] >= threshold)
			c_j = np.int8(x['synaptic_pev'][:,j,onset[i]] >= threshold)
			count[i,j] = np.sum(c_i*c_j)

	plt.figure()
	plt.imshow(count)
	plt.colorbar()
	plt.xticks(np.arange(5), [1,2,3,4,5])
	plt.yticks(np.arange(5), [1,2,3,4,5])
	plt.ylabel("Item")
	plt.xlabel("Item")
	plt.title('Count of Neurons Strongly Encoding for Each Pair of Items')
	plt.savefig('./pev_count/synaptic_{}_acc80.png'.format(threshold))