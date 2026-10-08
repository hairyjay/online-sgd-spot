import numpy as np
import os
import matplotlib.pyplot as plt

for path in os.scandir("price-trace"):
    print(path)
    if os.path.splitext(path)[-1].lower() == ".npy" and os.path.splitext(path)[-2].endswith("b_L"):
        with open(path, 'rb') as f:
            a = np.load(f)
            for i in range(1, a.shape[1]):
                a[1, i] += a[1, i-1]
            plt.plot(a[1, :], a[0, :])

plt.show()
