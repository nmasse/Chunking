import numpy as np
from parameters import *
import pickle
import stimulus
import matplotlib.pyplot as plt

x = pickle.load(open('./savedir/analysis_chunking_8_cue_on.pkl', 'rb'))

slices = [350, 420, 380, 520, 580, 600]

for i in range(len(slices)):
    print("\n\nAnalyzing weights for pulse "+str(i+1))
    arr = x['synaptic_pev'][:,i+1,slices[i]]
    w = []
    l = arr.argsort()[-4:][::-1]
    if i ==2:
        l = list(arr.argsort()[-2:][::-1])
        # l.remove(55)
        # l.remove(68)
    for ind in l:
        w.append(x['weights']['w_rnn'][ind][np.array(l)])
    print("Weights: ")
    print(w)
    print("Top 4 neurons: ", l)
    print("Average weight between 4 top neurons: ", np.mean(w))
    print("Average weight between all 100 neurons: ", np.mean(x['weights']['w_rnn']))
    print("Average weight between 80 excitatory neurons: ", np.mean(x['weights']['w_rnn'][:80,:80]))
    print("Average weight between 20 inhibitory neurons: ", np.mean(x['weights']['w_rnn'][80:,80:]))
