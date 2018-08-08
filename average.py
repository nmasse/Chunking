import analysis_new
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import os

def plot(filename,neuronal,synaptic,stability,syn_stability):
    plt.figure()
    plt.plot(np.mean(neuronal[0,0],axis=0), 'r')
    plt.plot(np.mean(synaptic[0,0],axis=0), 'g')
    plt.plot([0,375],[1/8,1/8],"k--")
    plt.savefig('./savedir/plots_new/'+filename[:-4]+'.png')
    plt.close()

    plt.figure()
    plt.imshow(np.mean(stability[0,0],axis=0))
    plt.colorbar()
    plt.savefig('./savedir/plots_new/'+filename[:-4]+'_heatmap_neuronal.png')
    plt.close()

    plt.figure()
    plt.imshow(np.mean(syn_stability[0,0],axis=0))
    plt.colorbar()
    plt.savefig('./savedir/plots_new/'+filename[:-4]+'_heatmap_synaptic.png')
    plt.close()


if __name__ == "__main__":
    outputs = os.listdir('./savedir/test_new')
    files = []
    files2 = []
    for filename in outputs:
        print(filename[:12])
        if filename[:12] == 'delay_type_0':
            files.append(filename)
        else:
            files2.append(filename)
    
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    if '.DS_Store' in files2:
        files2.remove('.DS_Store')

    results = []
    results2 = []
    for filename in files:
        f = open(('./savedir/test_new/'+filename),'rb')
        results.append(pickle.load(f))
    for filename in files2:
        f = open(('./savedir/test_new/'+filename),'rb')
        results2.append(pickle.load(f))

    a,b,c,d = results[0]['neuronal_sample_decoding'], results[0]['synaptic_sample_decoding'], results[0]['neuronal_sample_decoding_stability'], results[0]['synaptic_sample_decoding_stability']
    for i in range(1,len(results)):
        a += results[i]['neuronal_sample_decoding']
        b += results[i]['synaptic_sample_decoding']
        c += results[i]['neuronal_sample_decoding_stability']
        d += results[i]['synaptic_sample_decoding_stability']
    a /= len(results)
    b /= len(results)
    c /= len(results)
    d /= len(results)
    plot('type_0.pkl', a,b,c,d)

    a2,b2,c2,d2 = results2[0]['neuronal_sample_decoding'], results2[0]['synaptic_sample_decoding'], results2[0]['neuronal_sample_decoding_stability'], results2[0]['synaptic_sample_decoding_stability']
    for i in range(1,len(results2)):
        a2 += results2[i]['neuronal_sample_decoding']
        b2 += results2[i]['synaptic_sample_decoding']
        c2 += results2[i]['neuronal_sample_decoding_stability']
        d2 += results2[i]['synaptic_sample_decoding_stability']
    a2 /= len(results2)
    b2 /= len(results2)
    c2 /= len(results2)
    d2 /= len(results2)
    plot('type_1.pkl', a2,b2,c2,d2)

    plot('diff.pkl', a2-a, b2-b, c2-c, d2-d)

        
