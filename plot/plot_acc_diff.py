import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import json

timestamp = "20210620082307"

def get_acc_trace(timestamp):
    with open(os.path.join(timestamp, 'ts.npy'), 'rb') as f:
        a = np.load(f)
        return a

ax = plt.gca()
i = 0
thr = []
data = []
for run in os.scandir(os.path.join('runs/paper/uniform_rate_fixed_pricing')):
    if os.path.isdir(run):
        if os.path.exists(os.path.join(run, "stats.json")):

            color = next(ax._get_lines.prop_cycler)['color']
            a = get_acc_trace(run)
            data.append(a[:, 1])
            print(a[0, :])
            plt.plot(a[1:, 0], np.diff(a[:, 1]), color=color)

            #with open(os.path.join(run, "stats.json")) as json_file:
                #data = json.load(json_file)
                #threshold = data["target_itr"]
                #thr.append(threshold)
                #plt.vlines(threshold, ymin=0, ymax=100, color=color, linestyle='dashed', label="threshold @ {} iterations".format(threshold))
            i += 1


thr = np.array(thr)
#print(np.mean(thr), np.var(thr), np.std(thr))

plt.xlabel('Batches Arrived $\\frac{J}{K}$')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of model in relation to number of batches arrived')
#plt.grid(True)
plt.xlim(0, 300000)
#plt.legend()
plt.show()
