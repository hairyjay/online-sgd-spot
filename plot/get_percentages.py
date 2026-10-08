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

def get_override_price_trace(file):
    with open(file, 'rb') as f:
        a = np.load(f)
        return a

def get_batch_times(timestamp):
    with open(os.path.join(timestamp, 'ps.npy'), 'rb') as f:
        a = np.load(f)
        b = np.load(f)
        c = np.load(f)
        return c

def get_traces(struct, folder, path, deadline, availability, rate, adap, color, sm_marker, lg_marker, od_price=0.286, name=None, label=None, price_override=None):
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
    for run in os.scandir(os.path.join(folder, path)):
        if os.path.isdir(run):
            with open(os.path.join(run, "stats.json")) as json_file:
                file = json.load(json_file)

                threshold = file["target_itr"]
                data['thresholds'].append(threshold)

                acc = get_acc_trace(run)
                acc_time = np.zeros((acc.shape[0], 5))
                acc_time[:, :acc.shape[1]] = acc
                times = get_batch_times(run)
                if "scale" in file["rate_dist"]:
                    times = times * file["rate_dist"]["scale"]
                for t in range(acc_time.shape[0]):
                    if acc_time[t, 0] < times[-1, 0]:
                        acc_time[t, 3] = times[np.where(times[:, 0] == acc_time[t, 0]) , 1]

                price = get_price_trace(run)
                if price_override:
                    new_price = get_override_price_trace(price_override)
                    if "scale" in file["rate_dist"]:
                        new_price[1, :] /= file["rate_dist"]["scale"]
                    else:
                        new_price[1, :] /= 2000
                #print(path)
                #print(price)

                j = 1
                k = 0
                total_price = 0
                for t in range(acc_time.shape[0]):
                    while acc_time[t, 3] > price[j, 0]:
                        if price_override:
                            if price[j, 0] > new_price[1, k]:
                                k += 1
                                if k >= new_price.shape[1]:
                                    k = 0
                                    new_price[1, :] += price[j, 0]
                        od = N - price[j, 2] if data['availability'] < 1 else N
                        if price_override:
                            total_price += (od * od_price + (price[j, 3] - od) * new_price [0, k]) * (price[j, 0] - price[j-1, 0])
                        else:
                            total_price += (od * od_price + (price[j, 3] - od) * price[j, 1]) * (price[j, 0] - price[j-1, 0])
                        j += 1
                    acc_time[t, 4] = total_price / 3600

                if price_override:
                    mean_spot_price.append(np.mean(new_price[0, :]))
                else:
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
    if availability >= 1:
        data['exp_price'] = 1
    else:
        data['exp_price'] = np.max([deadline + (((availability*mean_spot_price - od_price)*(deadline-1)) / ((1-availability)*od_price)), mean_spot_price/od_price])

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

def get_data(folder, rate, od_price=0.286, price_override=None):
    struct = {}
    get_traces(struct, folder, 'ondemand_fixed', 1, 1, rate, False, 'darkorange', 'x', 'X', label='on demand', od_price=od_price, price_override=price_override)
    get_traces(struct, folder, '105_90_{}'.format(rate), 1.05, 0.9, rate, False, 'blue', '.', 'o', od_price=od_price, price_override=price_override)
    get_traces(struct, folder, 'adap_105_90_{}'.format(rate), 1.05, 0.9, rate, True, 'red', '.', 'o', od_price=od_price, price_override=price_override)
    return struct

def plot_cost(data, rate):
    def plot_cat(name):
        print(name, data[name]['mean_price']/data['ondemand_fixed']['mean_price'], data[name]['exp_price'])

    plot_cat('ondemand_fixed')
    plot_cat('105_90_{}'.format(rate))
    plot_cat('adap_105_90_{}'.format(rate))

def plot_acc(data, rate="fixed", adap=False, legend=False):
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
        plt.hlines( 85,
                    xmin=0, xmax=200, alpha=0.2,
                    color="lightgrey")
        plt.vlines( data[name]['mean_price'],
                    ymin=0, ymax=200, alpha=0.5,
                    color=data[name]['color'])
        plt.plot(   data[name]['mean_price'], 85,
                    color=data[name]['color'],
                    marker=data[name]['lg_marker'],
                    linestyle=linestyle,
                    label=data[name]['label'])
        plt.plot(   avg[:, 1], avg[:, 0],
                    color=data[name]['color'],
                    linestyle=linestyle)
        print(name, data[name]['mean_price'])

    adap_prefix = "adap_" if adap else ""
    plt.figure(figsize=(3.5, 2))
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.18)
    plt.ylabel('Test set accuracy (%)')
    plt.xlabel('Cost ($)')
    plot_line('ondemand_fixed'.format(rate), linestyle='solid')
    plot_line('{}105_80_{}'.format(adap_prefix, rate), linestyle='dashed')
    plot_line('{}110_80_{}'.format(adap_prefix, rate), linestyle='dashed')
    plot_line('{}105_90_{}'.format(adap_prefix, rate), linestyle='dashed')
    plt.xlim(0, 35)
    plt.ylim(0, 90)
    if legend:
        plt.legend()
    plt.savefig('../2026_acc_{}{}.pdf'.format(adap_prefix, rate))
    #plt.show()

def plot_loss(data, rate="fixed", adap=False):
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

    adap_prefix = "adap_" if adap else ""
    plt.figure(figsize=(3.5, 2))
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.18)
    plt.ylabel('Test set loss')
    plt.xlabel('Cost ($)')
    plot_line('ondemand_fixed', linestyle='solid')
    plot_line('{}105_80_{}'.format(adap_prefix, rate), linestyle='dashed')
    plot_line('{}110_80_{}'.format(adap_prefix, rate), linestyle='dashed')
    plot_line('{}105_90_{}'.format(adap_prefix, rate), linestyle='dashed')
    plt.yscale('log')
    plt.xlim(0, 35)
    plt.grid(which='both', alpha=0.5)
    plt.legend()
    plt.savefig('../2026_loss_{}{}.pdf'.format(adap_prefix, rate))
    plt.show()

price_override = "./price-trace/c5_us-west-2a_S.npy"
lower_price_override = "./price-trace/c5_us-west-2a_L.npy"

print('EMNIST FIXED')
data = get_data('../runs/runs_log/paper/', 'fixed', od_price=0.226, price_override=price_override)
plot_cost(data, 'fixed')
# print(data["ondemand_fixed"]["mean_time"])
# print(np.mean(data["ondemand_fixed"]["thresholds"]))

print('EMNIST UNIFORM')
data = get_data('../runs/runs_log/paper/', 'uniform', od_price=0.226, price_override=price_override)
plot_cost(data, 'uniform')

print('EMNIST FIXED ADAM')
data = get_data('../runs/runs_log/infocom2023-adam/', 'fixed', od_price=0.226, price_override=price_override)
plot_cost(data, 'fixed')

print('IMNIST FIXED')
data = get_data('../runs/runs_log/infocom2023-imnist/', 'fixed', od_price=0.226, price_override=price_override)
plot_cost(data, 'fixed')

print('IMNIST UNIFORM')
data = get_data('../runs/runs_log/infocom2023-imnist/', 'uniform', od_price=0.226, price_override=price_override)
plot_cost(data, 'uniform')

print('CIFAR FIXED')
data = get_data('../runs/runs_log/infocom2023-cifar/', 'fixed', od_price=0.226, price_override=price_override)
plot_cost(data, 'fixed')

print('CIFAR UNIFORM')
data = get_data('../runs/runs_log/infocom2023-cifar/', 'uniform', od_price=0.226, price_override=price_override)
plot_cost(data, 'uniform')