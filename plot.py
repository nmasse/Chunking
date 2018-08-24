import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from parameters import *

def plot_pev_cross_time(x, num_pulses, cue, pev_type):
    nrows = num_pulses//2 if num_pulses%2==0 else num_pulses
    ncols = 2 if num_pulses%2==0 else 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(15,10))
    i = 0
    for ax in axes.flat:
        im = ax.imshow(x[pev_type][:,i,:],aspect='auto')
        #im = ax.imshow(np.mean(x[pev_type][:,i,:,:],axis=2), aspect='auto')
        i += 1
    cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(im, cax=cax, **kw)
    plt.savefig("./savedir/"+pev_type+"_cross_time_"+str(num_pulses)+"_pulses_"+cue+".png")

def plot_pev_after_stim(x, num_pulses, cue, pev_type,time_lapse):
    nrows = num_pulses//2 if num_pulses%2==0 else num_pulses
    ncols = 2 if num_pulses%2==0 else 1

    plt.figure()
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,12))
    i = 0
    for ax in axes.flat:
        #im = ax.imshow(np.mean(x[pev_type][:,:,np.unique(x['timeline'])[2*i+1]+time_lapse,:],axis=2),aspect='auto')
        #eolongd = (par['dead_time']+par['fix_time'] + num_pulses * par['sample_time'] + (num_pulses-1)*par['delay_time'] + par['long_delay_time'])//par['dt']
        im = ax.imshow(x[pev_type][:,:,np.unique(x['timeline'])[2*i+1]+time_lapse],aspect='auto')
        i += 1
    cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(im, cax=cax, **kw)
    eolongd = (par['dead_time']+par['fix_time'] + num_pulses * par['sample_time'] + (num_pulses-1)*par['delay_time'] + par['long_delay_time'])//par['dt']
    #plt.imshow(x[pev_type][:,:,eolongd-time_lapse],aspect='auto')
    plt.savefig("./savedir/"+pev_type+"_after_stim_"+str(num_pulses)+"_pulses_"+cue+".png")
    plt.close()
    plt.figure()
    m = np.array(x[pev_type][:,:,eolongd-time_lapse])
    plt.hist(np.sum((m>0.1),axis=1), bins=range(9))
    plt.xticks(range(9))
    plt.xlim(xmin=0, xmax=9)
    plt.savefig("./savedir/"+pev_type+"_selectivity_"+str(num_pulses)+"_pulses_"+cue+".png")
    plt.close()

def shufffle_pev(x, num_pulses, cue, pev_type, acc_type, time_lapse):
    print('shuffle')
    print(np.unique(np.array(x['timeline'])))
    test_onset = [np.unique(np.array(x['timeline']))[-2*p-2] for p in range(num_pulses)][::-1]
    nrows = num_pulses//2 if num_pulses%2==0 else num_pulses
    ncols = 2 if num_pulses%2==0 else 1

    eolongd = (par['dead_time']+par['fix_time'] + num_pulses * par['sample_time'] + (num_pulses-1)*par['delay_time'] + par['long_delay_time'])//par['dt']

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15,12))
    i = 0
    for ax in axes.flat:
        #ax.scatter(x[pev_type][:,i,eolongd-1], np.mean(x[acc_type][i,:,:],axis=1), s=5)
        ax.scatter(x[pev_type][:,i,eolongd-1,0], np.mean(x[acc_type][i,:,:],axis=1), s=5)
        i += 1
    plt.savefig("./savedir/"+pev_type+"_shuffle_"+str(num_pulses)+"_pulses_"+cue+".png")

if __name__ == "__main__":
    num_pulses = [4]
    cue_list = ['cue_off']
    pev_type = ['synaptic_pev', 'neuronal_pev']
    acc_type = ['accuracy_syn_shuffled','accuracy_neural_shuffled']

    for num_pulses in num_pulses:
        for cue in cue_list:
            for i in range(len(pev_type)):
                    x = pickle.load(open('./savedir/RF_cue_analysis_RF_cue_sequence_cue_sequence_4_all_RF_long_delacc90_ay.pkl','rb'))
                    #x = pickle.load(open('./savedir/analysis_'+str(num_pulses)+"_"+cue+".pkl", 'rb'))
                    update_parameters(x['parameters'])
                    plot_pev_cross_time(x, num_pulses, cue, pev_type[i])
                    #plot_pev_after_stim(x, num_pulses, cue, pev_type[i], 10)
                    #shufffle_pev(x, num_pulses, cue, pev_type[i], acc_type[i], 10)
