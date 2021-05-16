import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt

timestamp = "20210514072412"

def get_acc_trace(timestamp):
    run_folder = os.path.join('../', 'runs', timestamp)
    with open(os.path.join(run_folder, 'ts.npy'), 'rb') as f:
        a = np.load(f)
    return a

a_a = get_acc_trace("20210502170139")
a_b = get_acc_trace("20210503104623")
a_c = get_acc_trace("20210503125329")
a_d = get_acc_trace("20210513120643")

a_0 = get_acc_trace("20210514072412")
a_1 = get_acc_trace("20210514211841")
a_2 = get_acc_trace("20210515085609")

plt.plot(a_a[:, 0], a_a[:, 1])
plt.plot(a_b[:, 0], a_b[:, 1])
plt.plot(a_c[:, 0], a_c[:, 1])
plt.plot(a_d[:, 0], a_d[:, 1])

plt.plot(a_0[:, 0], a_0[:, 1])
plt.plot(a_1[:, 0], a_1[:, 1])
plt.plot(a_2[:, 0], a_2[:, 1])

plt.xlabel('Batches Arrived')
plt.ylabel('Probability')
plt.title('Accuracy')
plt.grid(True)
plt.show()
