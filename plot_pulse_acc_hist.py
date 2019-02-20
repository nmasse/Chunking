import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors



file = 'analysis_restart_no_var_delay_6_acc97.pkl'

x = pickle.load(open("./savedir_restart/"+file, 'rb'))


bupu = plt.get_cmap('BuPu')
cNorm = colors.Normalize(vmin=0, vmax=range(10)[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=bupu)

color_list = []
for i in range(10):
	colorVal = scalarMap.to_rgba(range(10)[i])
	color_list.append(colorVal)


parsed = np.array(x['model_performance']['pulse_accuracy'])[::1500]
base = np.zeros((1,6))
parsed = np.concatenate((base, parsed), axis=0)

plt.figure()
for i in range(8):
	plt.bar(np.arange(6), parsed[i+1, :]-parsed[i,:], color=color_list[i+2],bottom=parsed[i,:],label='Iteration {}'.format((i+1)*1500))
plt.xticks(np.arange(6), ['Item 1', 'Item 2','Item 3','Item 4','Item 5','Item 6'])



plt.title('Recall Accuracy For Each Item Thoughout Training')
plt.ylabel('Accuracy')
plt.subplots_adjust(right=0.7)
plt.legend(bbox_to_anchor=(1.04,1))

plt.show()