import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from parameters import *

def plot_pev_cross_time(x, num_pulses, cue, pev_type):
    nrows = num_pulses//2 if num_pulses%2==0 else num_pulses
    ncols = 2 if num_pulses%2==0 else 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    i = 0
    for ax in axes.flat:
        im = ax.imshow(x[pev_type][:,i,:],aspect='auto')
        i += 1
    cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(im, cax=cax, **kw)
    plt.savefig("./savedir/var_delay/"+pev_type+"_cross_time_"+str(num_pulses)+"_pulses_"+cue+".png")

def plot_pev_after_stim(x, num_pulses, cue, pev_type,time_lapse):
    nrows = num_pulses//2 if num_pulses%2==0 else num_pulses
    ncols = 2 if num_pulses%2==0 else 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    i = 0
    for ax in axes.flat:
        im = ax.imshow(x[pev_type][:,:,np.unique(x['timeline'])[2*i+1]+time_lapse],aspect='auto')
        # eolongd = (par['dead_time']+par['fix_time'] + num_pulses * par['sample_time'] + (num_pulses-1)*par['delay_time'] + par['long_delay_time'])//par['dt']
        # im = ax.imshow(x[pev_type][:,:,eolongd-time_lapse],aspect='auto')
        i += 1
    cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(im, cax=cax, **kw)
    #eolongd = (par['dead_time']+par['fix_time'] + num_pulses * par['sample_time'] + (num_pulses-1)*par['delay_time'] + par['long_delay_time'])//par['dt']
    #plt.imshow(x[pev_type][:,:,eolongd-time_lapse],aspect='auto')
    plt.savefig("./savedir/var_delay/"+pev_type+"_after_stim_"+str(num_pulses)+"_pulses_"+cue+".png")
    plt.close()
    # m = np.array(x[pev_type][:,:,eolongd-time_lapse])
    # plt.hist(np.sum((m>0.1),axis=1), bins=range(9))
    # plt.xticks(range(9))
    #plt.xlim(xmin=0, xmax=8)
    # plt.savefig("./savedir/var_delay/"+pev_type+"_selectivity_"+str(num_pulses)+"_pulses_"+cue+".png")
    # plt.close()

def shufffle_pev(x, num_pulses, cue, pev_type, acc_type, time_lapse):
    test_onset = [np.unique(np.array(x['timeline']))[-2*p-2] for p in range(num_pulses)][::-1]
    nrows = num_pulses//2 if num_pulses%2==0 else num_pulses
    ncols = 2 if num_pulses%2==0 else 1

    eolongd = (par['dead_time']+par['fix_time'] + num_pulses * par['sample_time'] + (num_pulses-1)*par['delay_time'] + par['long_delay_time'])//par['dt']

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    i = 0
    for ax in axes.flat:
        ax.scatter(x[pev_type][:,i,eolongd-1], np.mean(x[acc_type][i,:,:],axis=1), s=5)
        i += 1
    plt.tight_layout()
    plt.savefig("./savedir/var_delay/"+pev_type+"_shuffle_"+str(num_pulses)+"_pulses_"+cue+".png")

num_pulses = [8]
cue_list = ['cue_on']
pev_type = ['synaptic_pev','neuronal_pev']
acc_type = ['accuracy_syn_shuffled','accuracy_neural_shuffled']

for num_pulses in num_pulses:
    for cue in cue_list:
        for i in range(len(pev_type)):
                x = pickle.load(open('./savedir/var_delay/shuffling_var_delay_'+str(num_pulses)+"_"+cue+".pkl", 'rb'))
                plot_pev_cross_time(x, num_pulses, cue, pev_type[i])
                plot_pev_after_stim(x, num_pulses, cue, pev_type[i], 10)
                shufffle_pev(x, num_pulses, cue, pev_type[i], acc_type[i], 10)
