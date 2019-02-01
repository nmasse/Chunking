import pickle
import numpy as np
import matplotlib.pyplot as plt


filepath = "./savedir_restart/"

base = ['analysis_restart_no_var_delay_6_', 'analysis_restart_var_delay_6_', 'analysis_restart_no_var_delay_5_', 'analysis_restart_var_delay_5_']
addition = ['acc80.pkl', 'acc90.pkl', 'acc97.pkl']

for a in addition:
	for b in base:
		file = b + a
		x = pickle.load(open(filepath + file, 'rb'))
		num_pulse = b[-2]
		acc = a[-6:-4]
		model = b[17:-3]
		plt.figure()
		plt.plot(x['pulse_accuracy'])
		plt.title("Pulse accuracy for {} model {} pulses {} accuracy".format(model, num_pulse, acc))
		plt.savefig('./pulse/{}_{}_pulses_{}acc.png'.format(model, num_pulse, acc))
		plt.close()