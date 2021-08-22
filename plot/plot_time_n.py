import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import json

#type = "binomial"
type = "uniform"

def get_acc_trace(timestamp):
    with open(os.path.join(timestamp, 'ts.npy'), 'rb') as f:
        a = np.load(f)
        return a

def get_batch_times(timestamp):
    with open(os.path.join(timestamp, 'ps.npy'), 'rb') as f:
        a = np.load(f)
        b = np.load(f)
        c = np.load(f)
        return c

ax = plt.gca()
i = 0
thr = []
for run in os.scandir('runs/paper/uniform_rate_fixed_pricing'):
    if os.path.isdir(run):
        if os.path.exists(os.path.join(run, "stats.json")):

            with open(os.path.join(run, "stats.json")) as json_file:
                data = json.load(json_file)
                #pricing = data["pricing"]

                #if pricing is None:
                #preempt_type = data["preempt"]["distribution"]
                #if preempt_type == type:
                color = next(ax._get_lines.prop_cycler)['color']
                #threshold = data["target_itr"]

                acc = get_acc_trace(run)
                acc_time = np.zeros((acc.shape[0], acc.shape[1] + 1))
                acc_time[:, :2] = acc
                times = get_batch_times(run)
                for t in range(acc_time.shape[0]):
                    acc_time[t, 2] = times[np.where(times[:, 0] == acc_time[t, 0]) , 1]
                #print(acc_time[:5, :])

                threshold = -1
                for i in range(10, acc_time.shape[0]):
                    if np.mean(acc_time[i-10:i, 1]) >= 85:
                        threshold = acc_time[i-1, 2]
                        break
                if threshold > 0:
                    plt.vlines(threshold, ymin=0, ymax=90, color=color, linestyle='dashed', alpha=0.5)
                thr.append(threshold)
                #plt.vlines(acc_time[np.where(acc_time[:, 0] == threshold), 2], ymin=0, ymax=100, color=color, linestyle='dashed')
                plt.plot(acc_time[:, 2], acc_time[:, 1], color=color, label="$N = {}$".format(data["size"]))
            i += 1

plt.hlines(85, xmin=0, xmax=8500, linestyle='dashed', alpha=0.5)
mean_thr = np.mean(thr)
print(mean_thr)
plt.vlines(mean_thr, ymin=0, ymax=90, linestyle='dashed')

plt.xlabel('Wall-clock time (s)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy in wall-clock time for {} preemption'.format(type))
#plt.grid(True)
#plt.xlim(0, 150000) #BINOM
plt.xlim(0, 10000)
plt.ylim(0, 90)
#handles, labels = ax.get_legend_handles_labels()
#print(labels)
#BINOM
#s_labels = [labels[2], labels[0], labels[3], labels[4], labels[1]]
#s_handles = [handles[2], handles[0], handles[3], handles[4], handles[1]]
#UNIF
#ax.legend(s_handles, s_labels)
#plt.legend()
plt.show()
