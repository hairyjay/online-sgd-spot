import json
import numpy as np
import os

directory = os.fsencode('./')

traces = []
lens = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".json"):
        with open(filename) as json_file:
            data = json.load(json_file)
            trace = data["data"]
            print(len(trace), np.mean(trace))

            traces.append(trace)
            lens.append(len(trace))

min_len = min(lens)
traces = np.array([t[:min_len] for t in traces])
print(traces.shape, np.mean(np.sum(traces, axis=0))/4)
