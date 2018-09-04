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


#trial_types = [['sequence', 'sequence_cue', 'RF_detection', 'RF_cue']]
trial_types = [['RF_cue','sequence_cue']]
num_pulses = [6,8,10]
all_RFs = [False]

# NEXT TO RUN IS VAR_PULSES

for pulse in num_pulses:
    for trial_type in trial_types:
        for all_RF in all_RFs:

            print('Training network on {} task(s) with {} pulses (all_RF = {})'.format(trial_type, str(pulse), str(all_RF)))
            save_fn = '{}_p{}_100_neuron_v2.pkl'.format('_'.join(trial_type), str(pulse))
            print('Saving in:', save_fn)

            updates = {
                'save_fn'       : save_fn,
                'num_pulses'    : pulse,
                'num_RFs'       : pulse,
                'trial_type'    : trial_type,
                'all_RF'        : all_RF,
                'n_hidden'      : 100,
                'var_num_pulses': True,
            }

            update_parameters(updates)

            try_model()
            print('\n'*5)
