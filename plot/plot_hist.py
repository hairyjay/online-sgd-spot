import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt

timestamp = "20210514072412"
#timestamp = "20210514211841"
#timestamp = "20210515085609"
run_folder = os.path.join('../', 'runs', timestamp)

with open(os.path.join(run_folder, 'ps.npy'), 'rb') as f:
    a = np.load(f)
    b = np.load(f)
    c = np.load(f)

print(a.shape, b.shape, c.shape)
print(b[-5:, :])

update_interval = b[:, 1]#np.diff(c[:, 1])
#print(np.min(update_interval), np.max(update_interval))
R = np.mean(update_interval)
print(R)

n, bins, patches = plt.hist(update_interval, 1000, density=True)
mu, sigma = norm.fit(update_interval)
y = norm.pdf(bins, mu, sigma)
plt.plot(bins, y)
plt.vlines(np.min(update_interval), 0, 15)
plt.vlines(mu, 0, 15)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of Update Interarrival Time')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.xlim(0, 0.2)
plt.ylim(0, 15)
plt.grid(True)
plt.show()
