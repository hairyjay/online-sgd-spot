import numpy as np
import ray
import asyncio
import argparse
import time
from datetime import datetime
import os
import json

from . import price
from . import rates
from . import cifar
from . import emnist

##################################################################
# Start ray cluster
##################################################################

ray.init(address="auto")

parser = argparse.ArgumentParser(description='PyTorch K-sync SGD')
parser.add_argument('--name','-n', default=None, type=str, help='experiment name, used for saving results')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
#parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=32, type=int, help='batch size on each worker')
parser.add_argument('--t', default=0.008, type=float, help='mean inter-arrival time of an individual data point')
parser.add_argument('--K', default=5, type=int, help='number of batches per update')
parser.add_argument('--test', default=1000, type=int, help='number of batches per accuracy check')
parser.add_argument('--target', default=0, type=float, help='target accuracy')
parser.add_argument('--J', default=195000, type=int, help='target interations')
parser.add_argument('--d', default=5580, type=int, help='target deadline')
parser.add_argument('--size', default=8, type=int, help='number of workers')
parser.add_argument('--a', default=0.95, type=float, help='spot instance availability')
parser.add_argument('--save', '-s', default=True, action='store_true', help='whether save the training results')
parser.add_argument('--fixed', '-f', action='store_true', help='fixed or uniform pricing')
parser.add_argument('--adap', '-d', action='store_true', help='adaptive method')
parser.add_argument('--autoexit', '-e', action='store_true', help='whether to exit on its own')
args = parser.parse_args()

if __name__ == "__main__":

    if not args.name:
        now = datetime.now()
        args.name = now.strftime("%Y%m%d%H%M%S") # RUN ID
    stats = vars(args)

    # PRICING
    #pricing = price.TracePricing("ca-central-1b_S.npy", 0.286, scale=2000) # TRACE "ca-central-1b_S.npy"
    pricing = price.TracePricing("ca-central-1b_L.npy", 0.186, scale=2000) # TRACE "ca-central-1b_L.npy"
    #pricing = price.FixedPricing(0.286) # FIXED PRICE
    stats["pricing"] = pricing.get_stats()

    # ALLOCATION
    allocation = price.InstanceAllocation(args)
    stats["allocation"] = allocation.get_stats()

    # DATA RATES
    if args.fixed:
        rate_dist = rates.FixedRates(args.t)
    else:
        rate_dist = rates.UniformRates(args.t)
    stats["rate_dist"] = rate_dist.get_stats()
    print(stats)

    # INITIALIZE WORKERS AND PARAMETER SERVER
    #experiment = cifar.CIFARShards(args, pricing)
    experiment = emnist.EMNISTShards(args, pricing)
    print("servers launched")

    start_time = time.time()

    # RUN WORKERS AND OTHER REMOTE PROCESSES
    experiment.run(start_time, allocation, rate_dist=rate_dist)
    print("processes launched")

    if args.autoexit:
        logs, test_stats = experiment.autoexit()
        stats.update(test_stats)
    else:
        # CONTROLS
        while True:
            cmd = input(">>>")
            #KILL
            if cmd == 'k':
                test_stats = experiment.terminate()
                stats.update(test_stats)
                print("all actors terminated")
                break

        # ON TERMINATION
        logs = experiment.save_logs()

    assert logs is not False

    path = os.path.join('runs', args.name)
    if not os.path.exists(path):
        os.makedirs(path)

    for l in logs:
        if l is not None:
            print(l[0])
            with open(os.path.join(path, "{}.npy".format(l[0])), 'wb') as f:
                for i in range(1, len(l)):
                    np.save(f, l[i])
    with open(os.path.join(path, "stats.json"), 'w') as f:
        json.dump(stats, f)
    print("run {} terminated".format(args.name))
