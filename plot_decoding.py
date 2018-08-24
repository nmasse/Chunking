import pickle
import matplotlib.pyplot as plt
import numpy as np

filename = 'analysis_RF_cue_sequence_cue_4_all_RF_100_neuron_test_RF_cue_test_sequence_cue.pkl'
x = pickle.load(open('./savedir/'+filename,'rb'))

for i in range(x['parameters']['num_pulses']):
	print(i)
	plt.plot(np.mean(x['synaptic_decoding'][i],axis=0))

print(x['parameters']['n_hidden'])
print(x['parameters']['trial_type'])
plt.savefig('/savedir/synaptic_decoding_'+filename[:-4]+'.png')
plt.close()

for i in range(x['parameters']['num_pulses']):
	print(i)
	plt.plot(np.mean(x['neuronal_decoding'][i],axis=0))

plt.savefig('/savedir/neuronal_decoding_'+filename[:-4]+'.png')
plt.close()

plt.imshow(x['mean_h'],aspect='auto')
plt.colorbar()
plt.savefig('/savedir/mean_h_'+filename[:-4]+'.png')