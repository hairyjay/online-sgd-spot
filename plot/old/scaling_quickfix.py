import numpy as np
import os
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import json

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

def get_traces(path, od_price=0.286):
    N = 64
    for run in os.scandir(os.path.join('../runs/a-emnist/', path)):
        if os.path.isdir(run):
            with open(os.path.join(run, "stats.json")) as json_file:
                file = json.load(json_file)

                acc = get_acc_trace(run)
                acc_time = np.zeros((acc.shape[0], 5))
                acc_time[:, :acc.shape[1]] = acc
                times = get_batch_times(run)
                if "scale" in file["rate_dist"]:
                    times = times * file["rate_dist"]["scale"]
                print(times[-25:-1, :])
                print(acc[-5:-1, :])
                for t in range(acc_time.shape[0]):
                    if acc_time[t, 0] < times[-1, 0]:
                        acc_time[t, 3] = times[np.where(times[:, 0] == acc_time[t, 0]) , 1]

                price = get_price_trace(run)
                print(price[-10:-1, 0:3])
                #print(path)
                #print(price)

                j = 1
                total_price = 0
                for t in range(acc_time.shape[0]):
                    while acc_time[t, 3] > price[j, 0]:
                        od = N - price[j, 2]
                        total_price += (od * od_price + (price[j, 3] - od) * price[j, 1]) * (price[j, 0] - price[j-1, 0])
                        j += 1
                    acc_time[t, 4] = total_price / 3600

get_traces('adap_110_80_uniform')