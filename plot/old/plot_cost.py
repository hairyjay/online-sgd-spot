import numpy as np
import os
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import json

plt.figure(figsize=(8, 3))
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

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

def get_traces(path, color, sm_marker, lg_marker, od_price=0.286, name=None):
    mean_price = []
    mean_time = []
    for run in os.scandir(os.path.join(path)):
        if os.path.isdir(run):
            with open(os.path.join(run, "stats.json")) as json_file:
                data = json.load(json_file)

                threshold = data["target_itr"]

                acc = get_acc_trace(run)
                acc_time = np.zeros((acc.shape[0], acc.shape[1] + 2))
                acc_time[:, :2] = acc[:, :2]
                times = get_batch_times(run)
                for t in range(acc_time.shape[0]):
                    acc_time[t, 2] = times[np.where(times[:, 0] == acc_time[t, 0]) , 1]

                price = get_price_trace(run)
                #print(price[-5:, :])

                j = 1
                total_price = 0
                for t in range(acc_time.shape[0]):
                    while acc_time[t, 2] > price[j, 0]:
                        od = N - price[j, 2]
                        total_price += (od * od_price + (price[j, 3] - od) * price[j, 1]) * (price[j, 0] - price[j-1, 0])
                        j += 1
                    acc_time[t, 3] = total_price / 3600

                mean_price.append(acc_time[np.where(acc_time[:, 0] == threshold), 3])
                mean_time.append(acc_time[np.where(acc_time[:, 0] == threshold), 2])

                '''
                ax1.text(   acc_time[np.where(acc_time[:, 0] == threshold), 2],
                            acc_time[np.where(acc_time[:, 0] == threshold), 3] - 1,
                            "${:.2f}".format(acc_time[np.where(acc_time[:, 0] == threshold), 3][0][0]),
                            fontsize=10)
                ax1.vlines(acc_time[np.where(acc_time[:, 0] == threshold), 2], ymin=0, ymax=200, color=color, linestyle='dashed')
                ax1.hlines( acc_time[np.where(acc_time[:, 0] == threshold), 3],
                            xmin=-5000, xmax=25000,
                            color='grey', alpha=0.2, linestyle='dashed')
                ax1.plot(   acc_time[:, 2], acc_time[:, 3],
                            color=color, alpha=0.05,
                            label="{}".format(data["pricing"]["distribution"]))
                '''
                ax1.plot(   acc_time[np.where(acc_time[:, 0] == threshold), 2],
                            acc_time[np.where(acc_time[:, 0] == threshold), 3],
                            color=color, alpha=0.3, marker=sm_marker)

                '''
                ax2.vlines( acc_time[np.where(acc_time[:, 0] == threshold), 3],
                            ymin=-5000, ymax=200,
                            color=color, alpha=0.1, linestyle='dashed')
                '''
                ax2.plot(   acc_time[:, 3], acc_time[:, 1],
                            color=color, alpha=0.05,
                            label="{}".format(data["pricing"]["distribution"]))

    mean_price = np.mean(np.array(mean_price))
    mean_time = np.mean(np.array(mean_time))
    ax1.hlines( mean_price,
                xmin=-5000, xmax=mean_time,
                color=color, alpha=0.2)
    ax1.vlines( mean_time,
                ymin=-50, ymax=mean_price,
                color=color, alpha=0.2)
    ax1.plot(   mean_time, mean_price, color=color, alpha=1, marker=lg_marker, label=name)
    '''
    ax1.text(   mean_time,
                mean_price - 1,
                "${:.2f}, {:.0f}s".format(mean_price, mean_time),
                fontsize=10)
    '''

N = 64

ax = plt.gca()
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

rate = 'fixed'

get_traces('runs/paper/ondemand_{}'.format(rate), 'orange', 'x', 'X', name="on demand only")
get_traces('runs/paper/105_90_{}'.format(rate), 'blue', '.', 'o', name="1.05x, 90%")
get_traces('runs/paper/105_80_{}'.format(rate), 'green', '+', 'P', name="1.05x, 80%")
get_traces('runs/paper/110_80_{}'.format(rate), 'purple', '1', 'v', name="1.1x, 80%")

#get_traces('runs/paper/adap_105_90_{}'.format(rate), 'red')

#get_traces('runs/paper/ondemand_{}'.format(rate), 'brown', '.', 'o', od_price=0.186)
#get_traces('runs/paper/lower_105_90_{}'.format(rate), 'black', '.', 'o', od_price=0.186)
#get_traces('runs/paper/lower_adap_105_90_{}'.format(rate), 'grey', '.', 'o', od_price=0.186)

ax1.set_xlabel('Wall-clock time (s)')
ax1.set_ylabel('Cost ($)')
ax1.set_title('Cost vs Time')
ax2.set_xlabel('Cost ($)')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy vs Cost')
#plt.grid(True)
ax1.set_xlim(5500, 7500)
ax1.set_ylim(18, 36)
ax2.set_ylim(0, 90)
ax1.legend()
plt.show()
