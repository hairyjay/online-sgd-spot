import numpy as np
import os
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import json

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#type = "binomial"
type = "uniform"
colors = ['green', 'magenta', 'red', 'purple', 'blue']
order = [0, 2, 1, 3, 4]

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

def get_price_trace(timestamp):
    with open(os.path.join(timestamp, 'price.npy'), 'rb') as f:
        a = np.load(f)
        return a

fig, ax1 = plt.subplots(figsize=(3.5, 2.5))
ax2 = ax1.twinx()
fig.subplots_adjust(bottom=0.16, left=0.15, right=0.87)
idx = 0
thr = []

def add_plot(json_file):
    data = json.load(json_file)
    #pricing = data["pricing"]

    #if pricing is None:
    #preempt_type = data["preempt"]["distribution"]
    #if preempt_type == type:
    color = colors[idx]
    #threshold = data["target_itr"]
    drift_start = data["drift_start"]
    drift_end = data["drift_start"] + data["drift_time"]
    drift_time = data["drift_time"]
    print(drift_start, drift_end, drift_time)

    acc = get_acc_trace(run)
    acc_time = np.zeros((acc.shape[0], acc.shape[1] + 1))
    acc_time[:, :acc.shape[1]] = acc
    times = get_batch_times(run)
    for t in range(acc_time.shape[0]):
        acc_time[t, 2] = times[np.where(times[:, 0] == acc_time[t, 0]) , 1]
    #print(acc_time[:5, :])
    t_before = 0
    t_after = 0
    for t, time in enumerate(acc_time[:, 2]):
        if time <= drift_start:
            t_before = t
        if time <= drift_end:
            t_after = t
    print(t_before, t_after)
    print(acc_time[:, 2])
    b_maximum = np.max(acc_time[:int(t_before), 1])
    a_minimum = np.min(acc_time[int(t_before):, 1])
    delta = b_maximum - a_minimum
    print(delta)

    threshold = -1
    for i in range(10, acc_time.shape[0]):
        if np.mean(acc_time[i-10:i, 1]) >= 90:
            threshold = acc_time[i-1, 2]
            break
    # if threshold > 0:
    #     ax1.vlines(threshold, ymin=0, ymax=100, color=color, linestyle='dashed')
    thr.append(threshold)
    #plt.vlines(acc_time[np.where(acc_time[:, 0] == threshold), 2], ymin=0, ymax=100, color=color, linestyle='dashed')
    ax1.plot(acc_time[:, 2], acc_time[:, 1], color=color, label="$N = {}$".format(data["size"]))
    ax1.plot([drift_start, drift_end], [42.5-order[idx]*7.5, 42.5-order[idx]*7.5], color=color, label="$N = {}$".format(data["size"]))

    # price = get_price_trace(run)
    ax1.fill_between([drift_start, drift_end], [42.5-order[idx]*7.5, 42.5-order[idx]*7.5], color=color, label="$N = {}$".format(data["size"]), alpha=.1, linewidth=0.0)
    ax1.text(drift_end + 100, 40-order[idx]*7.5, "drift: {}s".format(drift_time), color=color)

    ax1.text(2750, 70 - 7.5*order[idx], f'$\Delta$: -{delta:.0f}pts', color=color)
    ax1.vlines(2200 + 100*order[idx], ymin=a_minimum, ymax=b_maximum, color=color)
    # ax1.arrow(1800 + 100*order[idx], b_maximum, 0, -delta, color=color, head_width=2)
    #ax1.annotate("", xytext=(1800 + 100*order[idx], b_maximum), xy=(1800 + 100*order[idx], a_minimum), color=color, arrowprops=dict(arrowstyle="->"))
    # ax2.text(threshold/2+250, price[-1, 2]+1.5, "$N_s$ = {}".format(int(price[-1, 2])))

for run in os.scandir('../runs/drift/110_90'):
    if os.path.isdir(run):
        if os.path.exists(os.path.join(run, "stats.json")):

            with open(os.path.join(run, "stats.json")) as json_file:
                add_plot(json_file)
            idx += 1

for run in os.scandir('../runs/drift/105_90'):
    if os.path.isdir(run):
        if os.path.exists(os.path.join(run, "stats.json")):

            with open(os.path.join(run, "stats.json")) as json_file:
                add_plot(json_file)
            idx += 1

#ax1.hlines(90, xmin=0, xmax=8500, linestyle='dashed', alpha=0.5)
mean_thr = np.mean(thr)
print(mean_thr)
#ax1.vlines(mean_thr, ymin=0, ymax=90, linestyle='dashed')

ax1.set_xlabel('Wall-clock time (s)')
ax1.set_ylabel('Accuracy (%)')
ax2.set_ylabel('')
#plt.title('Accuracy in wall-clock time for {} preemption'.format(type))
#plt.grid(True)
#plt.xlim(0, 150000) #BINOM
ax1.set_xlim(0, 4125)
ax2.set_xlim(0, 4125)
ax1.set_ylim(0, 100)
ax2.set_ylim(0, 100)
#handles, labels = ax1.get_legend_handles_labels()
#print(labels)
#BINOM
#s_labels = [labels[2], labels[0], labels[3], labels[4], labels[1]]
#s_handles = [handles[2], handles[0], handles[3], handles[4], handles[1]]
#UNIF
#ax1.legend(s_handles, s_labels)
#plt.legend()
plt.savefig('../plots/drift_new.pdf')
plt.show()
