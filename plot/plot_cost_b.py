import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import json

plt.figure(figsize=(12, 6))

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

runs = ["A_G_20210515223007", "A_F2_20210520045717"]
#runs = ["A_U_20210516104530", "A_F2_20210520045717"]
#runs = ["A_R_20210519224931", "A_F_20210520045717"]

ax = plt.gca()
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
for r in runs:
    path = os.path.join('..', 'runs', r)
    with open(os.path.join(path, "stats.json")) as json_file:
        data = json.load(json_file)

        color = next(ax._get_lines.prop_cycler)['color']
        threshold = data["target_itr"]

        acc = get_acc_trace(path)
        acc_time = np.zeros((acc.shape[0], acc.shape[1] + 2))
        acc_time[:, :2] = acc
        times = get_batch_times(path)
        for t in range(acc_time.shape[0]):
            acc_time[t, 2] = times[np.where(times[:, 0] == acc_time[t, 0]) , 1]

        price = get_price_trace(path)
        price_n = np.zeros((price.shape[0], price.shape[1] + 1))
        price_n[:, :2] = price
        price_n[:-1, 2] = np.diff(price[:, 0])
        price_n[-1, 2] = max(acc_time[-1, 2], price[-1, 0]) - price[-1, 0]
        print(price_n[:5, :])
        print(acc_time[:5, :])

        bid = data["bids"]["bid1"]

        j = 1 if price_n.shape[0] > 1 else 0
        p = 0
        for t in range(acc_time.shape[0]):
            while True:
                if j >= price_n.shape[0] - 1: break
                if price_n[j, 0] > acc_time[t, 2]: break
                if bid > price_n[j, 1]: p += price_n[j-1, 2] * data["size"] * price_n[j, 1] / 7200
                j += 1
            if bid >= price_n[j, 1]:
                acc_time[t, 3] = p + abs(acc_time[t, 2] - price[j-1, 0]) * data["size"] * price_n[j, 1] / 7200
            else:
                acc_time[t, 3] = p
        #print(acc_time[:5, :])

        ax1.text(acc_time[np.where(acc_time[:, 0] == threshold), 2], acc_time[np.where(acc_time[:, 0] == threshold), 3] - 0.5, "${0:.3g}".format(acc_time[np.where(acc_time[:, 0] == threshold), 3][0][0]), fontsize=10)
        #ax1.vlines(acc_time[np.where(acc_time[:, 0] == threshold), 2], ymin=0, ymax=200, color=color, linestyle='dashed')
        ax1.hlines(acc_time[np.where(acc_time[:, 0] == threshold), 3], xmin=-5000, xmax=25000, color='grey', linestyle='dashed')
        ax1.plot(acc_time[:, 2], acc_time[:, 3], color=color, label="{}".format(data["pricing"]["distribution"]))
        ax1.plot(acc_time[np.where(acc_time[:, 0] == threshold), 2], acc_time[np.where(acc_time[:, 0] == threshold), 3], color=color, marker='o')

        ax2.vlines(acc_time[np.where(acc_time[:, 0] == threshold), 3], ymin=-5000, ymax=200, color=color, linestyle='dashed')
        ax2.plot(acc_time[:, 3], acc_time[:, 1], color=color, label="{}".format(data["pricing"]["distribution"]))

ax1.set_xlabel('Wall-clock time (s)')
ax1.set_ylabel('Cost ($)')
ax1.set_title('Cost vs Time')
ax2.set_xlabel('Cost ($)')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy vs Cost')
#plt.grid(True)
#ax1.set_xlim(1, 20000)
ax1.set_xlim(1, 25000)
#ax1.set_ylim(0, 5)
ax1.set_ylim(0, 16)
ax2.set_ylim(0, 80)
ax2.legend()
plt.show()
