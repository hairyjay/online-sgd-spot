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
    data['thresholds'] = []

    N = 64
    for run in os.scandir(os.path.join('runs/paper/', path)):
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
                    acc_time[t, 3] = times[np.where(times[:, 0] == acc_time[t, 0]) , 1]

                price = get_price_trace(run)

                j = 1
                total_price = 0
                for t in range(acc_time.shape[0]):
                    while acc_time[t, 3] > price[j, 0]:
                        od = N - price[j, 2]
                        total_price += (od * od_price + (price[j, 3] - od) * price[j, 1]) * (price[j, 0] - price[j-1, 0])
                        j += 1
                    acc_time[t, 4] = total_price / 3600

                mean_spot_price.append(np.mean(price[j, 1]))
                mean_price.append(acc_time[np.where(acc_time[:, 0] == threshold), 4])
                mean_time.append(acc_time[np.where(acc_time[:, 0] == threshold), 3])
                data['traces'].append(acc_time)

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
                ax1.plot(   acc_time[np.where(acc_time[:, 0] == threshold), 3],
                            acc_time[np.where(acc_time[:, 0] == threshold), 4],
                            color=color, alpha=0.3, marker=sm_marker)
                ax2.vlines( acc_time[np.where(acc_time[:, 0] == threshold), 3],
                            ymin=-5000, ymax=200,
                            color=color, alpha=0.1, linestyle='dashed')
                ax2.plot(   acc_time[:, 4], acc_time[:, 2],
                            color=color, alpha=0.05,
                            label="{}".format(data["pricing"]["distribution"]))
                '''

    mean_spot_price = np.mean(np.array(mean_spot_price))
    mean_price = np.mean(np.array(mean_price))
    mean_time = np.mean(np.array(mean_time))
    data['mean_spot_price'] = mean_spot_price
    data['mean_price'] = mean_price
    data['mean_time'] = mean_time

    '''
    ax1.hlines( mean_price,
                xmin=-5000, xmax=mean_time,
                color=color, alpha=0.2)
    ax1.vlines( mean_time,
                ymin=-50, ymax=mean_price,
                color=color, alpha=0.2)
    ax1.plot(   mean_time, mean_price, color=color, alpha=1, marker=lg_marker, label=name)
    ax1.text(   mean_time,
                mean_price - 1,
                "${:.2f}, {:.0f}s".format(mean_price, mean_time),
                fontsize=10)
    '''
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

    get_traces(struct, 'adap_105_90_{}'.format(rate), 1.05, 0.9, rate, True, 'red', '+', 'D')

    get_traces(struct, 'ondemand_{}'.format(rate), 1, 1, rate, False, 'brown', '.', 'o', od_price=0.186, name='lower_ondemand_{}'.format(rate), label='on demand')
    get_traces(struct, 'lower_105_90_{}'.format(rate), 1.05, 0.9, rate, False, 'black', '.', 'o', od_price=0.186)
    get_traces(struct, 'lower_adap_105_90_{}'.format(rate), 1.05, 0.9, rate, True, 'grey', '.', 'o', od_price=0.186)

    rate = 'uniform'
    get_traces(struct, 'ondemand_{}'.format(rate), 1, 1, rate, False, 'darkorange', 'x', 'X', label='on demand')
    get_traces(struct, '105_90_{}'.format(rate), 1.05, 0.9, rate, False, 'blue', '.', 'o')
    get_traces(struct, '105_80_{}'.format(rate), 1.05, 0.8, rate, False, 'green', '+', 'P')
    get_traces(struct, '110_80_{}'.format(rate), 1.1, 0.8, rate, False, 'magenta', '1', 'v')

    get_traces(struct, 'adap_105_90_{}'.format(rate), 1.05, 0.9, rate, True, 'red', '+', 'D')

    return struct

def plot_cost(data, rate, ymin=18, ymax=40, xmin=5500, xmax=7500, adap=False, legend=True, width=3, height=2):
    def plot_cat(name):
        for a, thr in zip(data[name]['traces'], data[name]['thresholds']):
            plt.plot(   a[np.where(a[:, 0] == thr), 3],
                        a[np.where(a[:, 0] == thr), 4],
                        color=data[name]['color'], alpha=0.3,
                        marker=data[name]['sm_marker'])

        plt.hlines(     data[name]['mean_price'],
                        xmin=0, xmax=data[name]['mean_time'],
                        color=data[name]['color'], alpha=0.2)
        plt.vlines(     data[name]['mean_time'],
                        ymin=0, ymax=data[name]['mean_price'],
                        color=data[name]['color'], alpha=0.2)
        plt.plot(       data[name]['mean_time'],
                        data[name]['mean_price'],
                        color=data[name]['color'], alpha=1,
                        linestyle = 'None',
                        marker=data[name]['lg_marker'],
                        label=data[name]['label'])

    plt.figure(figsize=(width, height))
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.18)
    plt.xlabel('Wall-clock time (s)')
    plt.ylabel('Cost ($)')
    if adap:
        plot_cat('ondemand_{}'.format(rate))
        plot_cat('105_90_{}'.format(rate))
        plot_cat('adap_105_90_{}'.format(rate))
    else:
        plot_cat('ondemand_{}'.format(rate))
        plot_cat('105_90_{}'.format(rate))
        plot_cat('105_80_{}'.format(rate))
        plot_cat('110_80_{}'.format(rate))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    if legend:
        plt.legend(loc=4)
    if adap:
        plt.savefig('../{}_adap.pdf'.format(rate))
    else:
        plt.savefig('../{}.pdf'.format(rate))
    plt.show()

def plot_acc(data, legend=False):
    def plot_line(name, linestyle='solid'):
        acc = []
        for a in data[name]['traces']:
            acc.append(a[:, (1, 4)])
        thr = np.mean(data[name]['thresholds'])

        lens    = [a.shape[0] for a in acc]
        maxlen  = max(lens)
        a       = np.zeros((len(acc), maxlen, 2))
        mask    = np.arange(maxlen) < np.array(lens)[:, None]
        a[mask, :] = np.concatenate(acc)
        acc     = np.ma.array(a, mask=~np.stack((mask, mask), axis=-1))

        #min     = np.ma.min(data, axis=0)
        #max     = np.ma.max(data, axis=0)
        avg     = np.ma.mean(acc, axis=0)
        #x       = np.arange(0, maxlen*1000, 1000) + 1000

        plt.vlines( data[name]['mean_price'],
                    ymin=0, ymax=200, alpha=0.5,
                    color=data[name]['color'], linestyle='dashed')
        plt.plot(   data[name]['mean_price'], 85,
                    color=data[name]['color'],
                    marker=data[name]['lg_marker'],
                    linestyle=linestyle,
                    label=data[name]['label'])
        plt.plot(   avg[:, 1], avg[:, 0],
                    color=data[name]['color'],
                    linestyle=linestyle)

    rate = "fixed"
    plt.figure(figsize=(3, 2))
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.18)
    plt.ylabel('Test set accuracy (%)')
    plt.xlabel('Cost ($)')
    plot_line('ondemand_{}'.format(rate), linestyle='solid')
    plot_line('105_80_{}'.format(rate), linestyle='dashed')
    plot_line('110_80_{}'.format(rate), linestyle='dashed')
    plot_line('105_90_{}'.format(rate), linestyle='dashed')
    plt.ylim(0, 90)
    if legend:
        plt.legend()
    plt.savefig('../acc.pdf')
    plt.show()

def plot_loss(data):
    def plot_line(name, linestyle='solid'):
        acc = []
        for a in data[name]['traces']:
            acc.append(a[:, (2, 4)])
        thr = np.mean(data[name]['thresholds'])

        lens    = [a.shape[0] for a in acc]
        maxlen  = max(lens)
        a       = np.zeros((len(acc), maxlen, 2))
        mask    = np.arange(maxlen) < np.array(lens)[:, None]
        a[mask, :] = np.concatenate(acc)
        acc     = np.ma.array(a, mask=~np.stack((mask, mask), axis=-1))

        #min     = np.ma.min(data, axis=0)
        #max     = np.ma.max(data, axis=0)
        avg     = np.ma.mean(acc, axis=0)
        #x       = np.arange(0, maxlen*1000, 1000) + 1000

        plt.plot(   data[name]['mean_price'],
                    avg[np.searchsorted(avg[:, 1], data[name]['mean_price'], side="left"), 0],
                    color=data[name]['color'],
                    marker=data[name]['lg_marker'],
                    linestyle=linestyle,
                    label=data[name]['label'])
        plt.plot(   avg[:, 1], avg[:, 0],
                    color=data[name]['color'],
                    linestyle=linestyle)

    rate = "fixed"
    plt.figure(figsize=(3, 2))
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.18)
    plt.ylabel('Test set loss')
    plt.xlabel('Cost ($)')
    plot_line('ondemand_{}'.format(rate), linestyle='solid')
    plot_line('105_80_{}'.format(rate), linestyle='dashed')
    plot_line('110_80_{}'.format(rate), linestyle='dashed')
    plot_line('105_90_{}'.format(rate), linestyle='dashed')
    plt.yscale('log')
    plt.grid(which='both', alpha=0.5)
    plt.legend()
    plt.savefig('../loss.pdf')
    plt.show()

def plot_savings(data):
    labels = ['High spot price \n homogeneous', 'Low spot price \n homogeneous', 'High spot price \n heterogeneous']
    ondemand = [100, 100, 100]
    spot_reg = [100 * data['105_90_fixed']['mean_price'] / data['ondemand_fixed']['mean_price'],
                100 * data['lower_105_90_fixed']['mean_price'] / data['lower_ondemand_fixed']['mean_price'],
                100 * data['105_90_uniform']['mean_price'] / data['ondemand_uniform']['mean_price']]
    adap_reg = [100 * data['adap_105_90_fixed']['mean_price'] / data['ondemand_fixed']['mean_price'],
                100 * data['lower_adap_105_90_fixed']['mean_price'] / data['lower_ondemand_fixed']['mean_price'],
                100 * data['adap_105_90_uniform']['mean_price'] / data['ondemand_uniform']['mean_price']]
    max_cost = [100 * (data['105_90_fixed']['deadline'] + ((1 - (1/data['105_90_fixed']['deadline'])) * data['105_90_fixed']['deadline'] * (data['105_90_fixed']['availability'] * data['105_90_fixed']['mean_spot_price'] - data['105_90_fixed']['od_price']) / (data['105_90_fixed']['od_price'] * (1 - data['105_90_fixed']['availability'])))),
                100 * (data['lower_105_90_fixed']['deadline'] + ((1 - (1/data['lower_105_90_fixed']['deadline'])) * data['lower_105_90_fixed']['deadline'] * (data['lower_105_90_fixed']['availability'] * data['lower_105_90_fixed']['mean_spot_price'] - data['lower_105_90_fixed']['od_price']) / (data['lower_105_90_fixed']['od_price'] * (1 - data['lower_105_90_fixed']['availability']))))]

    #print(adap_reg)

    plt.figure(figsize=(5, 3))
    plt.subplots_adjust(bottom=0.15)
    width = 0.85
    x = np.arange(len(labels))
    r0 = plt.bar(x - 3 * width / 8, ondemand, width/4, label='on demand', color='darkorange')
    r1 = plt.bar(x - width / 8,     spot_reg, width/4, label='regular cost optimization', color='steelblue')
    r2 = plt.bar(x + width / 8,     adap_reg, width/4, label='adaptive cost optimization', color='lightcoral')
    r3 = plt.bar(x[:2] + 3 * width / 8, max_cost, width/4, label='cost upper bound', color='lightgrey')
    plt.ylabel('Cost ratio to on-demand only pricing')
    plt.xticks(x, labels)
    #plt.bar_label(r0, padding=1, fmt='%.0f%%')
    plt.bar_label(r1, padding=1, fmt='%.0f%%')
    plt.bar_label(r2, padding=1, fmt='%.0f%%')
    plt.bar_label(r3, padding=1, fmt='%.0f%%')
    plt.legend(loc=4)
    plt.ylim(0, 105)
    plt.savefig('../savings.pdf')
    plt.show()

data = get_data()

#print(100 * data['105_90_uniform']['mean_price'] / data['ondemand_uniform']['mean_price'])
#print(100 * data['105_80_uniform']['mean_price'] / data['ondemand_uniform']['mean_price'])
#print(100 * data['110_80_uniform']['mean_price'] / data['ondemand_uniform']['mean_price'])

#print(data['105_90_fixed']['mean_spot_price'])
#print(data['lower_105_90_fixed']['mean_spot_price'])

plot_savings(data)

plot_loss(data)
plot_acc(data)

plot_cost(data, 'fixed', legend=False)
plot_cost(data, 'uniform', ymin=10, ymax=40, xmax=9000, width=5)
#plot_cost(data, 'fixed', adap=True)
#plot_cost(data, 'uniform', ymin=10, ymax=50, adap=True)
