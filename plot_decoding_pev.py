import pickle
import numpy as np
import matplotlib.pyplot as plt
import stimulus
import matplotlib as mpl
import os

file = 'analysis_h_init_restart_var_delay_6_low_noise_tcm2_acc85.pkl'

os.makedirs('./'+file)

x = pickle.load(open("./savedir_restart/"+file, 'rb'))

# plot decoding
decoding_type = ['synaptic_sample_decoding', 'neuronal_sample_decoding']
for decoding in decoding_type:
	plt.figure()
	for i in range(x['parameters']['num_pulses']):
		plt.plot(np.mean(x[decoding][0],axis=1)[i])
	for n in list(x['onset']):
		plt.axvline(x=n, color='black',linestyle='dashed',linewidth=1)

	plt.savefig('./'+file+'/'+decoding+'.png')
	plt.close()

# plot pev
pev_type = ['synaptic_pev', 'neuronal_pev']

for pev in pev_type:
	fig, axes = plt.subplots(nrows=x['parameters']['num_pulses']//2, ncols=2,figsize=(12,8))
	i = 0
	for ax in axes.flat:
		im = ax.imshow(x[pev][:,i,:],aspect='auto')
		i += 1
	cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
	plt.colorbar(im, cax=cax, **kw)
	plt.savefig('./'+file+'/'+pev+'.png')
	plt.close()