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

def get_traces(struct, path, deadline, availability, rate, adap, color, sm_marker, lg_marker, od_price=0.286, name=None, label=None):
    data = {}
    data['path'] = path
    data['color'] = color
    data['sm_marker'] = sm_marker
    data['lg_marker'] = lg_marker
    data['od_price'] = od_price
    data['deadline'] = deadline
    data['availability'] = availability
    data['rate'] = rate
    data['adaptive'] = adap
    if label is None:
        if adap:
            data['label'] = "$\\alpha$={}, $\\theta/\\theta_0$={}, adaptive".format(availability, deadline)
        else:
            data['label'] = "$\\alpha$={}, $\\theta/\\theta_0$={}".format(availability, deadline)
    else:
        data['label'] = label

    mean_spot_price = []
    mean_price = []
    mean_time = []
    data['traces'] = []
    data['costs'] = []
    data['thresholds'] = []

    N = 64
    for run in os.scandir(os.path.join('../runs/a-emnist/', path)):
        if os.path.isdir(run):
            with open(os.path.join(run, "stats.json")) as json_file:
                file = json.load(json_file)

                threshold = file["target_itr"]
                data['thresholds'].append(threshold)

                acc = get_acc_trace(run)
                acc_time = np.zeros((acc.shape[0], 5))
                acc_time[:, :acc.shape[1]] = acc
                times = get_batch_times(run)
                for t in range(acc_time.shape[0]):
                    if acc_time[t, 0] < times[-1, 0]:
                        acc_time[t, 3] = times[np.where(times[:, 0] == acc_time[t, 0]) , 1]

                price_raw = get_price_trace(run)
                #print(path)
                #print(price)

                j = 1
                price = np.zeros((price_raw.shape[0], price_raw.shape[1]+1))
                price[:, :price_raw.shape[1]] = price_raw
                total_price = 0
                for t in range(acc_time.shape[0]):
                    while acc_time[t, 3] > price[j, 0]:
                        od = N - price[j, 2] if data['availability'] < 1 else N
                        total_price += (od * od_price + (price[j, 3] - od) * price[j, 1]) * (price[j, 0] - price[j-1, 0])
                        price[j, -1] = total_price
                        j += 1
                    acc_time[t, 4] = total_price / 3600

                mean_spot_price.append(np.mean(price[j, 1]))
                mean_price.append(acc_time[np.where(acc_time[:, 0] == threshold), 4])
                mean_time.append(acc_time[np.where(acc_time[:, 0] == threshold), 3])
                data['traces'].append(acc_time)
                data['costs'].append(price)

    mean_spot_price = np.mean(np.array(mean_spot_price))
    mean_price = np.mean(np.array(mean_price))
    mean_time = np.mean(np.array(mean_time))
    data['mean_spot_price'] = mean_spot_price
    data['mean_price'] = mean_price
    data['mean_time'] = mean_time

    if name is None:
        name = path
    struct[name] = data

def get_data():
    struct = {}

    rate = 'fixed'
    get_traces(struct, 'ondemand_{}'.format(rate), 1, 1, rate, False, 'darkorange', 'x', 'X', label='on demand')
    get_traces(struct, '105_90_{}'.format(rate), 1.05, 0.9, rate, False, 'blue', '.', 'o')
    get_traces(struct, '105_80_{}'.format(rate), 1.05, 0.8, rate, False, 'green', '+', 'P')
    get_traces(struct, '110_80_{}'.format(rate), 1.1, 0.8, rate, False, 'magenta', '1', 'v')

    #get_traces(struct, 'adap_105_90_{}'.format(rate), 1.05, 0.9, rate, True, 'cyan', '.', 'o')
    #get_traces(struct, 'adap_105_80_{}'.format(rate), 1.05, 0.8, rate, True, 'yellow', '+', 'P')
    #get_traces(struct, 'adap_110_80_{}'.format(rate), 1.1, 0.8, rate, True, 'red', '1', 'v')

    # get_traces(struct, 'ondemand_{}'.format(rate), 1, 1, rate, False, 'brown', '.', 'o', od_price=0.186, name='lower_ondemand_{}'.format(rate), label='on demand')
    # get_traces(struct, 'lower_105_90_{}'.format(rate), 1.05, 0.9, rate, False, 'black', '.', 'o', od_price=0.186)
    # get_traces(struct, 'lower_adap_105_90_{}'.format(rate), 1.05, 0.9, rate, True, 'grey', '.', 'o', od_price=0.186)

    rate = 'uniform'
    #get_traces(struct, 'ondemand_{}'.format(rate), 1, 1, rate, False, 'darkorange', 'x', 'X', label='on demand')
    get_traces(struct, '105_90_{}'.format(rate), 1.05, 0.9, rate, False, 'blue', '.', 'o')
    get_traces(struct, '105_80_{}'.format(rate), 1.05, 0.8, rate, False, 'green', '+', 'P')
    get_traces(struct, '110_80_{}'.format(rate), 1.1, 0.8, rate, False, 'magenta', '1', 'v')

    #get_traces(struct, 'adap_105_90_{}'.format(rate), 1.05, 0.9, rate, True, 'cyan', '.', 'o')
    #get_traces(struct, 'adap_105_80_{}'.format(rate), 1.05, 0.8, rate, True, 'yellow', '+', 'P')
    get_traces(struct, 'adap_110_80_{}'.format(rate), 1.1, 0.8, rate, True, 'red', '1', 'v')

    #get_traces(struct, 'failed/adap_105_90_{}'.format(rate), 1.05, 0.9, rate, True, 'cyan', '.', 'o')
    #get_traces(struct, 'failed/adap_105_80_{}'.format(rate), 1.05, 0.8, rate, True, 'yellow', '+', 'P')
    #get_traces(struct, 'failed/adap_110_80_{}'.format(rate), 1.1, 0.8, rate, True, 'red', '1', 'v')
    
    get_traces(struct, 'test', 1.1, 0.8, rate, True, 'red', '1', 'v')

    print(struct)
    return struct

def plot_cost(data, rate, ymin=0, ymax=40, xmin=0, xmax=7500, adap=False, legend=True, width=3, height=2):
    def plot_cat(name):
        for a, thr in zip(data[name]['costs'], data[name]['thresholds']):
            plt.plot(   a[:, 0],
                        a[:, 2],
                        color=data[name]['color'], alpha=0.01,
                        marker=data[name]['sm_marker'])

    plt.figure(figsize=(width, height))
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.18)
    plt.xlabel('Wall-clock time (s)')
    plt.ylabel('Cost ($)')
    #plot_cat('ondemand_fixed'.format(rate))
    #plot_cat('test'.format(rate))
    plot_cat('110_80_{}'.format(rate))
    plot_cat('adap_110_80_{}'.format(rate))
    #plt.xlim(xmin, xmax)
    #plt.ylim(ymin, ymax)
    if legend:
        plt.legend(loc=4)
    if adap:
        plt.savefig('../{}_adap.pdf'.format(rate))
    else:
        plt.savefig('../{}.pdf'.format(rate))
    plt.show()

data = get_data()

#print(100 * data['105_90_uniform']['mean_price'] / data['ondemand_uniform']['mean_price'])
#print(100 * data['105_80_uniform']['mean_price'] / data['ondemand_uniform']['mean_price'])
#print(100 * data['110_80_uniform']['mean_price'] / data['ondemand_uniform']['mean_price'])

#print(data['105_90_fixed']['mean_spot_price'])
#print(data['lower_105_90_fixed']['mean_spot_price'])

#plot_savings(data)

#plot_loss(data)
#plot_acc(data)

#plot_cost(data, 'fixed', legend=False)
#plot_cost(data, 'dirichlet', width=5)
#plot_cost(data, 'fixed', adap=True)
plot_cost(data, 'uniform', adap=True)
#plot_cost(data, 'uniform', ymin=10, ymax=50, adap=True)
