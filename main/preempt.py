import numpy as np
import numpy.random as random
import ray
import asyncio
import argparse
import time
from datetime import datetime
import os
import json

import torch
import torch.optim as optim
from torchvision import datasets, transforms

from nets import *
from data import *
from price import *

##################################################################
# Start ray cluster
##################################################################

ray.init(address="auto")

##################################################################
# parameter server
##################################################################

@ray.remote(num_cpus=4)
class ParameterServer(object):
    def __init__(self, price_distr, l=0.005, k=5, t=100):
        self.params = 0
        self.l = l
        self.k = k
        self.t = t
        self.queue = asyncio.Queue()
        self.processed = 0
        self.workers = None
        self.start_time = None

        self.arrival_time = []
        self.gradient_time = []
        self.update_time = []
        self.price_distr = price_distr
        self.price = self.price_distr.start_price()
        self.running = True
        self.price_log = []

        self.net = Net()
        print("param server init")

    def signal(self, worker_index, itr):
        #print("got signal from worker {} batch {}".format(worker_index, itr))
        self.queue.put_nowait((worker_index, itr))
        return True

    async def queue_processor(self, workers, test_server, start_time):
        self.workers = workers
        self.start_time = start_time

        while True:
            batches = []
            for i in range(self.k):
                b = await self.queue.get()
                if b == "stop":
                    arrival_time = np.array(self.arrival_time)
                    gradient_time = np.array(self.gradient_time)
                    update_time = np.array(self.update_time)
                    return 'ps', arrival_time, gradient_time, update_time
                batches.append(b)
                self.queue.task_done()

            group_start = time.time()

            weights = []
            for param in self.net.parameters():
                weights.append(param.data)
            w_ref = ray.put(weights)

            #tasks = [self.workers[b[0]].compute_gradients.remote(w_ref, b[1]) for b in batches]
            #grad = ray.get(tasks)
            grad = await asyncio.gather(*[self.workers[b[0]].compute_gradients.remote(w_ref, b[1]) for b in batches])
            #grad = await asyncio.gather(*[self.workers[b[0]].compute_gradients.remote(weights, b[1]) for b in batches])

            self.apply_gradients(grad)
            del batches, weights, w_ref, grad#, tasks
            self.arrival_time.append([self.processed, group_start - self.start_time])
            self.gradient_time.append([self.processed, time.time() - group_start])
            self.update_time.append([self.processed, time.time() - self.start_time])

            if self.processed % self.t == 0:
                #print("QUEUE SIZE: {}".format(self.queue.qsize()))
                self.queue_acc(test_server)

    def apply_gradients(self, gradients):
        grad = np.mean(gradients, axis = 0)
        for i, param in enumerate(self.net.parameters()):
            param.data -= self.l * torch.Tensor(grad[i])

        self.processed += self.k
        del grad
        #print(self.processed)

    def queue_acc(self, test_server):
        weights = []
        for param in self.net.parameters():
            weights.append(param.data)

        test_server.test_acc.remote(weights, self.processed)
        del weights

    async def price_generator(self, bid1, bid2=None, n2=0):
        if bid2:
            if bid2 < bid1:
                raise ValueError('bid2 cannot be lower than bid1')
        n1 = len(self.workers) - n2
        status1 = True
        status2 = True
        last_update = time.time() - self.start_time
        self.price_log.append([last_update, self.price])
        print("starting price set to {}".format(self.price))

        while self.running:
            if bid2:
                if bid2 < self.price and status2:
                    for w in self.workers[-n2:]:
                        w.preempt.remote()
                    status2 = False
                    print("n2 preempted due to pricing")
                elif bid2 >= self.price and not status2:
                    for w in self.workers[-n2:]:
                        w.restart.remote()
                    status2 = True
                    print("n2 restarted due to pricing")
            if bid1 < self.price and status1:
                for w in self.workers[:n1]:
                    w.preempt.remote()
                status1 = False
                print("n1 preempted due to pricing")
            elif bid1 >= self.price and not status1:
                for w in self.workers[:n1]:
                    w.restart.remote()
                status1 = True
                print("n1 restarted due to pricing")

            self.price, interval = self.price_distr.get_price()
            await asyncio.sleep(interval)
            last_update = time.time() - self.start_time
            self.price_log.append([last_update, self.price])
            print("price changed to {}".format(self.price))

        return 'price', np.array(self.price_log)

    def terminate(self):
        self.queue.put_nowait("stop")
        self.running = False

##################################################################
# test server
##################################################################

@ray.remote(num_cpus=4)
class TestServer(object):
    def __init__(self, testset, target=None):
        self.processed = 0
        self.weights = {}
        self.queue = asyncio.Queue()
        self.accuracy = []
        self.target = target
        self.target_itr = -1

        #dataset
        self.net = Net()
        self.test_loader = torch.utils.data.DataLoader(testset,
                                            batch_size=128,
                                            shuffle=False)
        print("test server init")

    def test_acc(self, weights, itr):
        self.weights[itr] = weights
        self.queue.put_nowait(itr)

    async def test_processor(self):
        while True:
            itr = await self.queue.get()
            self.queue.task_done()
            if itr == "stop":
                return 'ts', np.array(self.accuracy)

            for i, param in enumerate(self.net.parameters()):
                param.data = self.weights[itr][i]
            del self.weights[itr]

            self.processed = itr
            acc = self.get_acc()
            print("ACCURACY AFTER {} BATCHES: {}".format(self.processed, acc))
            self.accuracy.append([self.processed, acc])

            if self.target and len(self.accuracy) >= 10 and self.target_itr < 0:
                last_10_acc = np.mean(np.array([a[1] for a in self.accuracy[-10:]]))
                if last_10_acc > self.target * 100:
                    self.target_itr = self.processed
                    print("TARGET OF {}% REACHED AFTER {} BATCHES AT {}%".format(self.target*100, self.target_itr, last_10_acc))

    def get_acc(self):
        self.net.eval()
        top1 = AverageMeter()
        # correct = 0
        # total = 0
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            # inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = self.net(inputs)
            acc1 = comp_accuracy(outputs, targets)
            top1.update(acc1[0], inputs.size(0))
        return top1.avg

    def terminate(self):
        self.queue.put_nowait("stop")
        return {"target_itr": self.target_itr}

##################################################################
# worker
##################################################################

@ray.remote(num_cpus=4)
class Worker(object):
    def __init__(self, worker_index, ps, trainset, B=32, l=0.001, lr=0.03):
        self.worker_index = worker_index
        self.ps = ps
        self.curritr = 0
        self.batches = {}
        self.B = B
        self.l = l
        self.lr = lr

        self.running = True
        self.preempt = False
        self.arrival_time = []
        self.gradient_time = []

        self.train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=B,
                                            shuffle=True)
        self.iterator = iter(self.train_loader)
        self.net = Net()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.trainacc = AverageMeter()
        self.accs = []
        print("worker {} init".format(self.worker_index))

    async def batch_generator(self, start_time):
        while self.running:
            # SIMULATE MINI-BATCH INTER-ARRIVAL TIME
            wait_time = random.gamma(self.B, self.l)
            #print("wait time: {}".format(wait_time))
            await asyncio.sleep(wait_time)
            #print("WORKER {} BATCH {} ARRIVED".format(self.worker_index, self.curritr))

            # SIGNAL PARAMETER SERVER
            try:
                data, target = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.train_loader)
                data, target = next(self.iterator)

            if not self.preempt:
                self.batches[self.curritr] = (data, target)
                self.arrival_time.append([self.curritr, time.time() - start_time])

                self.ps.signal.remote(self.worker_index, self.curritr)
                self.curritr += 1

        arrival_time = np.array(self.arrival_time)
        gradient_time = np.array(self.gradient_time)
        return str(self.worker_index), arrival_time, gradient_time

    def compute_gradients(self, weights, itr):
        batch_start = time.time()

        for i, param in enumerate(self.net.parameters()):
            param.data = weights[i]

        data, target = self.batches[itr]
        # data, target = data.cuda(), target.cuda()
        output = self.net(data)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()

        grads = []
        for param in self.net.parameters():
            grads.append(param.grad.data.numpy())

        del self.batches[itr], data, target, output, loss
        self.gradient_time.append([itr, time.time() - batch_start])
        #print("worker {} gradient of batch {} computed".format(self.worker_index, itr))
        return grads

    def preempt(self):
        self.preempt = True

    def restart(self):
        self.preempt = False

    def get_preempted(self):
        return self.preempt

    def terminate(self):
        self.running = False;

parser = argparse.ArgumentParser(description='PyTorch K-sync SGD')
parser.add_argument('--name','-n', default=None, type=str, help='experiment name, used for saving results')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
#parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=32, type=int, help='batch size on each worker')
parser.add_argument('--l', default=0.005, type=float, help='arrival rate of an individual data point')
parser.add_argument('--K', default=5, type=int, help='number of batches per update')
parser.add_argument('--test', default=1000, type=int, help='number of batches per accuracy check')
parser.add_argument('--target', default=0.7, type=float, help='target accuracy')
parser.add_argument('--size', default=8, type=int, help='number of workers')
parser.add_argument('--R', default=8, type=int, help='number of returned workers')
parser.add_argument('--save', '-s', default=True, action='store_true', help='whether save the training results')
args = parser.parse_args()

if __name__ == "__main__":

    # Initialize datalogging
    if not args.name:
        now = datetime.now()
        args.name = now.strftime("%Y%m%d%H%M%S") # RUN ID
    stats = vars(args)

    # Pricing Model
    #pricing = Uniform_Pricing(0.2, 1, 10) # UNIFORM SYNTHETIC
    pricing = Gaussian_Pricing(0.6, 0.175, 10) # GAUSSIAN SYNTHETIC
    stats["pricing"] = pricing.get_stats()

    #bids
    bids = {"bid1": 0.303030} # ONE BID
    #bids = {"bid1": 10, "bid2": 100, "n2": 2} # TWO BIDS
    stats["bids"] = bids

    # Initialize workers and parameter server
    testset = get_testset()
    ps = ParameterServer.remote(pricing, k=args.K, t=args.test)
    ts = TestServer.remote(testset, target=args.target)
    workers = []
    for i in range(args.size):
        trainset = partition_dataset(i, args.size, args.bs)
        workers.append(Worker.remote(i, ps, trainset, B=args.bs, l=args.l, lr=args.lr))
    print("servers launched")

    start_time = time.time()

    # Run workers and other remote processes
    processes = []
    processes.append(ps.queue_processor.remote(workers, ts, start_time))
    processes.append(ps.price_generator.remote(**bids))
    processes.append(ts.test_processor.remote())
    for w in workers:
        processes.append(w.batch_generator.remote(start_time))
    print("processes launched")

    # Main loop
    print("waiting for the end of time")
    while True:
        cmd = input(">>>")

        #KILL
        if cmd == 'k':
            for w in workers:
                w.terminate.remote()
            ps.terminate.remote()
            test_stats = ray.get(ts.terminate.remote())
            stats.update(test_stats)
            print("all actors terminated")
            break
        '''
        #PREEMPT ALL WORKERS
        elif cmd == 's':
            for w in workers:
                w.preempt.remote()
        #RESTART ALL WORKERS
        elif cmd == 'r':
            for w in workers:
                w.restart.remote()
        '''

    # Runs on termination
    path = os.path.join('runs', args.name)
    if not os.path.exists(path):
        os.makedirs(path)
    for p in processes:
        logs = ray.get(p)
        print(logs[0])
        with open(os.path.join(path, "{}.npy".format(logs[0])), 'wb') as f:
            for i in range(1, len(logs)):
                np.save(f, logs[i])
    with open(os.path.join(path, "stats.json"), 'w') as f:
        json.dump(stats, f)
    print("run {} terminated".format(args.name))
