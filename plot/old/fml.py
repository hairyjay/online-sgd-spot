import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import json

file = os.path.join('../', 'runs', 'a-emnist ondemand', '20240822121254', 'ps.npy')
with open(file, 'rb') as f:
    data = np.load(f)
    #data[0][1] = 1.0
    print(data[-1])
#with open(file, 'wb') as f:
    #np.save(f, data)
