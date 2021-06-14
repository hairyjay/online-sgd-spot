import numpy as np
import ray
import asyncio
import argparse
import time
from datetime import datetime
import os
import json

import price
import cifar

##################################################################
# Start ray cluster
##################################################################

ray.init(address="auto")

parser = argparse.ArgumentParser(description='PyTorch K-sync SGD')
parser.add_argument('--name','-n', default=None, type=str, help='experiment name, used for saving results')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
#parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=32, type=int, help='batch size on each worker')
parser.add_argument('--t', default=0.005, type=float, help='mean inter-arrival time of an individual data point')
parser.add_argument('--K', default=5, type=int, help='number of batches per update')
parser.add_argument('--test', default=1000, type=int, help='number of batches per accuracy check')
parser.add_argument('--target', default=0.65, type=float, help='target accuracy')
parser.add_argument('--size', default=8, type=int, help='number of workers')
parser.add_argument('--save', '-s', default=True, action='store_true', help='whether save the training results')
args = parser.parse_args()

if __name__ == "__main__":

    # Initialize datalogging
    if not args.name:
        now = datetime.now()
        args.name = now.strftime("%Y%m%d%H%M%S") # RUN ID
    stats = vars(args)

    # Pricing Model
    #pricing = price.Uniform_Pricing(0.2, 1, 10) # UNIFORM SYNTHETIC
    #pricing = price.Gaussian_Pricing(0.6, 0.175, 10) # GAUSSIAN SYNTHETIC
    #pricing = price.Trace_Pricing("ca-central-1b_S.npy", scale=500) # TRACE "ca-central-1b_S.npy"
    pricing = price.Fixed_Pricing(0.286) # FIXED PRICE
    stats["pricing"] = pricing.get_stats()

    #bids
    #bids = {"bid1": 0.675377277376705} # ONE BID - GAUSSIAN SYNTHETIC
    #bids = {"bid1": 0.7333333333333334} # ONE BID - UNIFORM SYNTHETIC
    #bids = {"bid1": 0.1606} # ONE BID - TRACE "ca-central-1b_S.npy"
    bids = {"bid1": 0.286} # FIXED PRICE
    #bids = {"bid1": 10, "bid2": 100, "n2": 2} # TWO BIDS
    stats["bids"] = bids
    print(stats)

    # Initialize workers and parameter server
    #ps = ParameterServer.remote(Net, pricing, k=args.K, t=args.test)
    #ts = TestServer.remote(Net)
    #workers = []
    #for i in range(args.size):
    #    workers.append(Worker.remote(i, args.size, ps, Net, B=args.bs, lr=args.lr))
    experiment = cifar.CIFARShards(args, pricing)
    print("servers launched")

    start_time = time.time()

    # Run workers and other remote processes
    #processes = []
    #processes.append(ps.queue_processor.remote(workers, ts, start_time))
    #processes.append(ps.price_generator.remote(**bids))
    #testset = get_testset()
    #processes.append(ts.test_processor.remote(get_testset, target_acc=args.target))
    #for i, w in enumerate(workers):
    #    processes.extend([w.input_generator.remote(partition_dataset, t=args.t), w.queue_processor.remote(start_time)])
    experiment.run(start_time, bids)
    print("processes launched")

    # Main loop
    print("waiting for the end of time")
    while True:
        cmd = input(">>>")

        #KILL
        if cmd == 'k':
            #for w in workers:
            #    w.terminate.remote()
            #ps.terminate.remote()
            #test_stats = ray.get(ts.terminate.remote())
            test_stats = experiment.terminate()
            stats.update(test_stats)
            print("all actors terminated")
            break

    # Runs on termination
    logs = experiment.save_logs()
    path = os.path.join('runs', args.name)
    if not os.path.exists(path):
        os.makedirs(path)

    for l in logs:
        if l is not None:
            print(l[0])
            with open(os.path.join(path, "{}.npy".format(l[0])), 'wb') as f:
                for i in range(1, len(l)):
                    np.save(f, l[i])
    #for p in processes:
        #logs = ray.get(p)
        #if logs is not None:
            #print(logs[0])
            #with open(os.path.join(path, "{}.npy".format(logs[0])), 'wb') as f:
                #for i in range(1, len(logs)):
                    #np.save(f, logs[i])
    with open(os.path.join(path, "stats.json"), 'w') as f:
        json.dump(stats, f)
    print("run {} terminated".format(args.name))
