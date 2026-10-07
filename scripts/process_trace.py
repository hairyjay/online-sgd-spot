import json
import numpy as np
from datetime import datetime
import os

with open('../price-trace/price_trace_us_w2_c5_2026.json') as json_file:
    data = json.load(json_file)
    trace = data["SpotPriceHistory"]
    data = {}
    for t in trace:
        if (t["AvailabilityZone"], t["ProductDescription"]) not in data:
            dt = datetime.fromisoformat("2021-05-10T08:00:00")
            data[(t["AvailabilityZone"], t["ProductDescription"])] = [[], [dt.timestamp()]]
        log = data[(t["AvailabilityZone"], t["ProductDescription"])]
        log[0].insert(0, float(t["SpotPrice"]))
        dt = datetime.fromisoformat(t["Timestamp"])
        #print(dt.timestamp())
        log[1].insert(0, dt.timestamp())
        #print(price, dt, times)

    #print(data.keys())
    for d in data:
        price = np.array(data[d][0])
        time = np.diff(np.array(data[d][1]))
        #print(data[d][1][-1] - data[d][1][0])
        #print(d, len(data[d][0]), len(data[d][1]))
        log = np.array([price, time])
        #print(log.shape)
        print(d[1][0], np.mean(log[0, :]), np.min(log[0, :]), np.max(log[0, :]))
        with open(os.path.join("..", "price-trace", "c5_{}_{}.npy".format(d[0], d[1][0])), 'wb') as f:
            np.save(f, log)
