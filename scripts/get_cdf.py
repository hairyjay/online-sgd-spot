import numpy as np
import os

filename = "ca-central-1b_S.npy"
filepath = os.path.join("..", "main", "price-trace", filename)

with open(filepath, 'rb') as f:
    data = np.load(f)

totaltime = np.sum(data[1, :])
#print(totaltime)
print(data)

sorted = data[data[:, 1].argsort()]
#print(sorted)

cdf = sorted
for i in range(1, cdf.shape[1]):
    cdf[1, i] += cdf[1, i-1]
totaltime -= cdf[1, 0]
cdf[1, :] -= cdf[1, 0]
cdf[1, :] /= totaltime

print(cdf, totaltime)


time_delay = 1.5

for i in range(0, cdf.shape[1]):
    if 1/1.5 < cdf[1, i]:
        print(cdf[:, i-1])
        break
