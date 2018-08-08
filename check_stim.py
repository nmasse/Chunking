import stimulus
from parameters import *

stim = stimulus.Stimulus()
par['trial_type'] = 'DMSvar'
par['batch_size'] = 20
par['rule'] = 3
par['check_stim'] = True
trial_info = stim.generate_trial(test_mode=False)

        # if par['check_stim']:
        #     for i in range(10):
        #         plt.figure()
        #         plt.title("num puslse:")
        #         plt.imshow(trial_info['desired_output'][:,:,i],aspect='auto')
        #         plt.show()
        #         plt.close()
        #         plt.figure()
        #         plt.imshow(trial_info['neural_input'][:,:,i],aspect='auto')
        #         plt.show()
        #         plt.close()