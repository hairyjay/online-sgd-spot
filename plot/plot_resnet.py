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

fig, ax1 = plt.subplots(figsize=(4.5, 3.5))
ax2 = ax1.twinx()
fig.subplots_adjust(bottom=0.16, left=0.13, right=0.87)
ax1.hlines( 85,
            xmin=0, xmax=66150, alpha=0.6,
            color="lightgrey")
ax1.vlines(66150, ymin=0, ymax=100, alpha=1, color="lightgrey", linestyle='dashed')
n = 0
thr = []
colors = ['orange', 'blue', 'magenta']
labels = ["on demand", "$\\alpha = 0.9, \\theta/\\theta_0=1.05$", "$\\alpha = 0.9, \\theta/\\theta_0=1.05$, adap"]
for run in os.scandir('../runs/imagenet'):
    if os.path.isdir(run):
        if os.path.exists(os.path.join(run, "stats.json")):

            with open(os.path.join(run, "stats.json")) as json_file:
                print(run)
                data = json.load(json_file)
                #pricing = data["pricing"]

                #if pricing is None:
                #preempt_type = data["preempt"]["distribution"]
                #if preempt_type == type:
                color = colors[n]
                threshold = data["target_itr"]

                acc = get_acc_trace(run)
                acc_time = np.zeros((acc.shape[0], acc.shape[1] + 1))
                acc_time[:, :acc.shape[1]] = acc
                times = get_batch_times(run)
                for t in range(acc_time.shape[0]):
                    acc_time[t, 2] = times[np.where(times[:, 0] == acc_time[t, 0]) , 1]
                #print(acc_time[:5, :])
                # threshold = -1
                # for i in range(10, acc_time.shape[0]):
                #     if np.mean(acc_time[i-10:i, 1]) >= 90:
                #         threshold = acc_time[i-1, 2]
                #         break
                if threshold > 0:
                    ax1.vlines(threshold, ymin=0, ymax=100, color=color, linestyle='dashed')
                thr.append(threshold)
                target_time = acc_time[np.where(acc_time[:, 0] == threshold), 2]


                price = get_price_trace(run)
                if data['a'] >= 1.0:
                    total_cost = target_time*data['size']*data['pricing']['on_demand']/3600
                    # ax2.plot([0, target_time[0, 0]], [0, total_cost[0, 0]], color=color, label="$N = {}$".format(data["size"]))
                    ax2.fill_between([0, target_time[0, 0]], [0, total_cost[0, 0]], color=color, alpha=.1, linewidth=0.0)
                    ax2.plot(target_time[0, 0], total_cost[0, 0], color=color, alpha=1, marker='o', label=labels[n], linestyle = 'None')
                    ax2.hlines(total_cost[0, 0], xmin=target_time[0, 0], xmax=75000, color=color)
                    ax2.text(67500, total_cost[0, 0] + 10, "${:.0f}".format(total_cost[0, 0]), color=color)
                    ax2.text(52500, total_cost[0, 0] + 10, "{:.0f}s".format(target_time[0, 0]), color=color)
                else:
                    t = 0
                    for i in range(price.shape[0]):
                        if price[i, 0] >= target_time:
                            t = i
                            break
                    print(t)
                    price[:, 4] /= 3600
                    # ax2.plot(price[:t+1, 0], price[:t+1, 4], color=color, label="$N = {}$".format(data["size"]))
                    ax2.fill_between(price[:t+1, 0], price[:t+1, 4], color=color, alpha=.1, linewidth=0.0)
                    ax2.plot(price[t, 0], price[t, 4], color=color, alpha=1, marker='o', label=labels[n], linestyle = 'None')
                    ax2.hlines(price[t, 4], xmin=price[t, 0], xmax=75000, color=color)
                    if n < 2:
                        ax2.text(67500, price[t, 4] + 10, "${:.0f}".format(price[t, 4]), color=color)
                    else:
                        ax2.text(67500, price[t, 4] - 25, "${:.0f}".format(price[t, 4]), color=color)
                    ax2.text(52500, 100 + 25*n, "{:.0f}s".format(target_time[0, 0]), color=color)
                # ax2.text(150, price[0, 2]-5, "$N_s$ = {}".format(int(price[0, 2])))
                # ax2.text(threshold/2+250, price[-1, 2]+1.5, "$N_s$ = {}".format(int(price[-1, 2])))

                ax1.vlines(acc_time[np.where(acc_time[:, 0] == threshold), 2], ymin=0, ymax=100, color=color, linestyle='dashed')
                # ax1.plot(acc_time[:, 2], acc_time[:, 1], color=color, label="$N = {}$".format(data["size"]), alpha=0.3)
                ax1.plot(acc_time[:, 2], acc_time[:, 3], color=color, label=labels[n])
            n += 1

#ax1.hlines(90, xmin=0, xmax=8500, linestyle='dashed', alpha=0.5)
mean_thr = np.mean(thr)
print(mean_thr)
#ax1.vlines(mean_thr, ymin=0, ymax=90, linestyle='dashed')

ax1.set_xlabel('Wall-clock time (s)')
ax1.set_ylabel('Top-5 Accuracy (%)')
ax2.set_ylabel('Cost of training ($)')
#plt.title('Accuracy in wall-clock time for {} preemption'.format(type))
#plt.grid(True)
#plt.xlim(0, 150000) #BINOM
ax1.set_xlim(0, 75000)
ax2.set_xlim(0, 75000)
ax1.set_ylim(0, 100)
ax2.set_ylim(0, 400)
#handles, labels = ax1.get_legend_handles_labels()
#print(labels)
#BINOM
#s_labels = [labels[2], labels[0], labels[3], labels[4], labels[1]]
#s_handles = [handles[2], handles[0], handles[3], handles[4], handles[1]]
#UNIF
ax2.legend(loc=4)
#plt.legend()
plt.savefig('../2026_imagenet.pdf')
# plt.show()
