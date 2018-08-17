import numpy as np
from parameters import *
import matplotlib.pyplot as plt


class Stimulus:

    def __init__(self):

        # Generate tuning functions
        self.create_tuning_functions()

    def generate_trial(self, task):

        if task == "sequence_one":
            return self.generate_sequence_one_trial()

        elif task == "sequence_all":
            return self.generate_sequence_all_trial()

        elif task == "RF_detection":
            return self.generate_RF_detection_trial()

        elif task == "RF_cue":
            return self.generate_RF_cue_trial()

        else:
            return None

    def generate_sequence_one_trial():
        return None

    def generate_sequence_all_trial():
        return None

    def generate_RF_detection_trial():
        return None

    def generate_RF_cue_trial():
        return None

    def create_tuning_functions(self):

        # Motion tuning   --> directional preferences
        # Fixation tuning --> just fixation
        # Rule tuning     --> current task

        motion_tuning = np.zeros([par['num_motion_dirs'], par['num_RFs'], par['n_input']])
        fix_tuning    = np.zeros([par['n_input'], 1])
        rule_tuning   = np.zeros([par['n_input'], 1])

        # Generate lists of preferred and possible stimulus directions
        pref_dirs = np.float32(np.arange(0,2*np.pi,2*np.pi/par['num_motion_tuned']))
        stim_dirs = np.float32(np.arange(0,2*np.pi,2*np.pi/par['num_motion_dirs']))

        # Tune individual neurons to specific stimulus directions
        for n in range(par['num_motion_tuned']):
            diff = np.cos(stim_dirs-pref_dirs[n])
            for r in range(par['num_RFs']):
                motion_tuning[:,r,r*par['num_motion_tuned']+n] = par['tuning_height']*np.exp(par['kappa']*diff)/np.exp(par['kappa'])

        # Tune fixation neurons to the correct height
        for n in range(par['num_fix_tuned']):
            fix_tuning[par['total_motion_tuned']+n,0] = par['tuning_height']

        # Tune rule neurons to the correct height
        for n in range(par['num_rule_tuned']):
            rule_tuning[par['total_motion_tuned']+par['num_fix_tuned']+n,0] = par['tuning_height']

        # Set tunings to class elements
        self.motion_tuning = motion_tuning
        self.fix_tuning    = fix_tuning
        self.rule_tuning   = rule_tuning



s = Stimulus()
