import numpy as np
from parameters import *
import model
import sys
import pickle
import historian

code_state = historian.record_code_state()

def try_model():

    try:
        if len(sys.argv) > 1:
            print('Running on GPU {}.'.format(sys.argv[1]))
            model.main(gpu_id=sys.argv[1], code_state=code_state)
        else:
            print('Running on CPU.')
            model.main(code_state=code_state)

    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')

"""
# pulse5 = 'debug_sequence_sequence_cue_RF_detection_RF_cue_p5_v0acc80_regenerated_acc80.pkl'
# pulse6 = 'debug_sequence_sequence_cue_RF_detection_RF_cue_p6_v1acc70_.pkl'

pulse4 = 'RF_cue_sequence_cue_p4_100_neuron_high_lr_v0_acc95.pkl'
pulse5 = 'RF_cue_sequence_cue_p5_100_neuron_high_lr_v0_acc90.pkl'


filename = pulse5
data = pickle.load(open('./savedir/'+filename, 'rb'))
data['parameters']['save_fn'] = filename[:-10] + '.pkl'#'_regenerated.pkl'
data['parameters']['weight_load_fn'] = './savedir/' + filename
data['parameters']['load_prev_weights'] = True

update_parameters(data['parameters'])
par['h_init'] = data['weights']['h_init']

print('Model now starting: {} pulses, {} RFs'.format(par['num_pulses'], par['num_RFs']))
try_model()
quit()
"""






#trial_types = [['sequence', 'sequence_cue', 'RF_detection', 'RF_cue']]
trial_types = [['RF_cue','sequence_cue']]
num_pulses = [5,7]
all_RFs = [False]

for pulse in num_pulses:
    for trial_type in trial_types:
        for all_RF in all_RFs:

            print('Training network on {} task(s) with {} pulses (all_RF = {})'.format(trial_type, str(pulse), str(all_RF)))
            save_fn = '{}_p{}_100_relu_input.pkl'.format('_'.join(trial_type), str(pulse))
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
