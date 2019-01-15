import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

# filename = 'RF_cue_analysis_RF_cue_sequence_cue_sequence_4_all_RF_long_delacc90_ay.pkl'
# x = pickle.load(open('./savedir/'+filename,'rb'))

files = os.listdir('./analysis_results/')
for filename in files:
    print("PLOTTING FILE ", filename)
    x = pickle.load(open('./analysis_results/' + filename, 'rb'))

    for i in range(x['parameters']['num_pulses']):
    	print(i)
    	plt.plot(np.mean(x['sequence']['synaptic_decoding'][i],axis=0))

    print(x['parameters']['n_hidden'])
    print(x['parameters']['trial_type'])
    plt.savefig('./plots/'+filename[:-4]+'/synaptic_decoding_'+filename[:-4]+'.png')
    plt.close()

    for i in range(x['parameters']['num_pulses']):
    	print(i)
    	plt.plot(np.mean(x['sequence']['neuronal_decoding'][i],axis=0))

    plt.savefig('./plots/'+filename[:-4]+'/neuronal_decoding_'+filename[:-4]+'.png')
    plt.close()

    plt.imshow(x['sequence']['mean_h'],aspect='auto')
    plt.colorbar()
    plt.savefig('./plots/'+filename[:-4]+'/mean_h_'+filename[:-4]+'.png')
