import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import json

timestamp = "20210620082307"

def get_time_trace(timestamp):
    with open(os.path.join(timestamp, 'ps.npy'), 'rb') as f:
        a = np.load(f)
        return a

ax = plt.gca()
i = 0
batches = []
times = []
for run in os.scandir(os.path.join('..', 'runs', 'a-emnist ondemand')):
    if os.path.isdir(run):
        if os.path.exists(os.path.join(run, "stats.json")):

            color = 'blue'#next(ax._get_lines.prop_cycler)['color']
            a = get_time_trace(run)
            plt.plot(a[:, 0], a[:, 1], color=color)

            with open(os.path.join(run, "stats.json")) as json_file:
                data = json.load(json_file)
                threshold = data["target_itr"]
                #plt.vlines(threshold, ymin=0, ymax=5000, color=color, linestyle='dashed', label="threshold @ {} iterations".format(threshold))
                for r in a:
                    if r[0] == threshold:
                        print(threshold, r[1], threshold/r[1], threshold/(r[1]*64*256))
                        batches.append(threshold)
                        times.append(r[1])
            i += 1

print(np.mean(batches), np.mean(times), np.mean(batches)/np.mean(times))
#print(np.mean(thr), np.var(thr), np.std(thr))

#plt.xlabel('Batches Arrived $\\frac{J}{K}$')
#plt.ylabel('Accuracy (%)')
#plt.title('Accuracy of model in relation to number of batches arrived')
#plt.grid(True)
#plt.xlim(0, 800000)
#plt.ylim(0, 90)
#plt.legend()
#plt.show()
