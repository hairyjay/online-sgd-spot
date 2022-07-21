import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import json

file = os.path.join('../', 'runs', 'A_F2_20210520045717', 'price.npy')
with open(file, 'rb') as f:
    data = np.load(f)
    #data[0][1] = 1.0
    print(data)
#with open(file, 'wb') as f:
    #np.save(f, data)
