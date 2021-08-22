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
            #print(a[0, :])
            #plt.plot(a[:, 0], a[:, 1], color=color)

            threshold = -1
            for i in range(10, a.shape[0]):
                if np.mean(a[i-10:i, 1]) >= 85:
                    threshold = a[i-1, 0]
                    break
            if threshold > 0:
                plt.vlines(threshold, ymin=0, ymax=90, color=color, linestyle='dashed', alpha=0.5)
            thr.append(threshold)

            #with open(os.path.join(run, "stats.json")) as json_file:
                #data = json.load(json_file)
                #threshold = data["target_itr"]
                #thr.append(threshold)
                #plt.vlines(threshold, ymin=0, ymax=100, color=color, linestyle='dashed', alpha=0.5, label="threshold @ {} iterations".format(threshold))
            i += 1
lens    = [len(d) for d in data]
maxlen  = max(lens)
a       = np.zeros((len(data), maxlen))
mask    = np.arange(maxlen) < np.array(lens)[:, None]
a[mask] = np.concatenate(data)
data    = np.ma.array(a, mask=~mask)

min     = np.ma.min(data, axis=0)
max     = np.ma.max(data, axis=0)
avg     = np.ma.mean(data, axis=0)
x       = np.arange(0, maxlen*1000, 1000) + 1000
plt.fill_between(x, min, max, color='r', alpha=0.4, linewidth=0)
plt.plot(x, avg, color='r')

print(np.mean(thr), np.var(thr), np.std(thr))
plt.vlines(np.mean(thr), ymin=0, ymax=100, linestyle='dashed')

plt.xlabel('Batches Arrived $\\frac{J}{K}$')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of model in relation to number of batches arrived')
#plt.grid(True)
plt.xlim(0, 300000)
plt.ylim(0, 90)
#plt.legend()
plt.show()
