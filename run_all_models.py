import numpy as np
from parameters import *
import model
import sys
import pickle

def try_model():

    try:
        if len(sys.argv) > 1:
            print('Running on GPU {}.'.format(sys.argv[1]))
            model.main(sys.argv[1])
        else:
            print('Running on CPU.')
            model.main()

    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')


trial_types = ['sequence', 'sequence_cue', 'RF_detection', 'RF_cue']
trial_types = [['RF_cue', 'sequence_cue']]
num_pulses = [4,6,8,10]
all_RFs = [False]

for pulse in num_pulses:
    for trial_type in trial_types:
        for all_RF in all_RFs:

            if not all_RF and 'sequence' != trial_type:
                continue

            print('Training network on {} task(s) with {} pulses (all_RF = {})'.format(trial_type, str(pulse), str(all_RF)))
            if all_RF:
                save_fn = '{}_{}_all_RF_100_neuron.pkl'.format('_'.join(trial_type), str(pulse))
            else:
                save_fn = '{}_{}_one_RF_100_neuron.pkl'.format('_'.join(trial_type), str(pulse))

            updates = {
                'save_fn'       : save_fn,
                'num_pulses'    : pulse,
                'num_RFs'       : pulse,
                'trial_type'    : trial_type,
                'all_RF'        : all_RF,
                'n_hidden'      : 100,
            }

            if 'sequence_cue' in trial_type:
                updates['resp_cue_time'] = 500
            else:
                updates['resp_cue_time'] = 200

            update_parameters(updates)

            try_model()
            print('\n'*5)
