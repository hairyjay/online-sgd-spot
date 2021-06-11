import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import json

plt.figure(figsize=(12, 6))

#type = "binomial"
type = "uniform"

def get_acc_trace(timestamp):
    with open(os.path.join(timestamp, 'ts.npy'), 'rb') as f:
        a = np.load(f)
        return a

def get_price_trace(timestamp):
    with open(os.path.join(timestamp, 'price.npy'), 'rb') as f:
        a = np.load(f)
        return a

def get_batch_times(timestamp):
    with open(os.path.join(timestamp, 'ps.npy'), 'rb') as f:
        a = np.load(f)
        b = np.load(f)
        c = np.load(f)
        return c

ax = plt.gca()
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
i = 0
for run in os.scandir(os.path.join('../', 'runs')):
    if os.path.isdir(run):
        if os.path.exists(os.path.join(run, "stats.json")):

            with open(os.path.join(run, "stats.json")) as json_file:
                data = json.load(json_file)
                pricing = data["pricing"]

                if pricing is None:
                    preempt_type = data["preempt"]["distribution"]
                    if preempt_type == type:
                        color = next(ax._get_lines.prop_cycler)['color']
                        threshold = data["target_itr"]

                        acc = get_acc_trace(run)
                        acc_time = np.zeros((acc.shape[0] + 1, acc.shape[1] + 2))
                        acc_time[1:, :2] = acc
                        times = get_batch_times(run)
                        for t in range(1, acc_time.shape[0]):
                            acc_time[t, 2] = times[np.where(times[:, 0] == acc_time[t, 0]) , 1]

                        price = get_price_trace(run)
                        price_n = np.zeros((price.shape[0], price.shape[1] + 1))
                        price_n[:, :2] = price
                        price_n[:-1, 2] = np.diff(price[:, 0])
                        price_n[-1, 2] = max(acc_time[-1, 2], price[-1, 0]) - price[-1, 0]
                        print(data["size"])
                        print(price_n[:5, :])

                        j = 0
                        p = 0
                        for t in range(1, acc_time.shape[0]):
                            while True:
                                if j >= price_n.shape[0] - 1: break
                                if price_n[j, 0] > acc_time[t, 2]: break
                                p += price_n[j-1, 2] * price_n[j, 1] / 7200
                                j += 1
                            acc_time[t, 3] = p + abs(acc_time[t, 2] - price[j-1, 0]) * price_n[j, 1] / 7200

                        ax1.hlines(acc_time[np.where(acc_time[:, 0] == threshold), 3], xmin=0, xmax=200000, color=color, linestyle='dashed')
                        ax1.plot(acc_time[:, 2], acc_time[:, 3], color=color, label="$N = {}$".format(data["size"]))
                        ax1.plot(acc_time[np.where(acc_time[:, 0] == threshold), 2], acc_time[np.where(acc_time[:, 0] == threshold), 3], color=color, marker='o')

                        ax2.vlines(acc_time[np.where(acc_time[:, 0] == threshold), 2], ymin=0, ymax=100, color=color, linestyle='dashed')
                        ax2.plot(acc_time[:, 2], acc_time[:, 1], color=color, label="$N = {}$".format(data["size"]))
            i += 1

ax1.set_xlabel('Wall-clock time (s)')
ax1.set_ylabel('Cost ($)')
ax1.set_title('Cost vs Time')
ax2.set_xlabel('Wall-clock time (s)')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy vs Time')
#plt.grid(True)
#ax1.set_xlim(0, 140000) #BINOM
ax1.set_xlim(0, 100000) #UNIF
ax1.set_ylim(0, 20)
#ax1.set_ylim(0, 40)
ax2.set_ylim(0, 80)
handles, labels = ax1.get_legend_handles_labels()
print(labels)
#BINOM
#s_labels = [labels[3], labels[1], labels[4], labels[0], labels[2]]
#s_handles = [handles[3], handles[1], handles[4], handles[0], handles[2]]
#UNIF
s_labels = [labels[2], labels[0], labels[3], labels[4], labels[1]]
s_handles = [handles[2], handles[0], handles[3], handles[4], handles[1]]
ax1.legend(s_handles, s_labels)
plt.show()
