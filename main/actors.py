import numpy as np
import ray
import asyncio
import time

import torch
import torch.optim as optim
#import torch.distributed as dist
#import torch.multiprocessing as mp

from . import data_tools
#from . import price

##################################################################
# parameter server
##################################################################

@ray.remote(num_cpus=4)
class ParameterServer(object):
    def __init__(self, Net, ts, price_distr, size, lr=0.005, k=5, t=100, B=256):
        self.params = 0
        self.lr = lr
        self.k = k
        self.t = t
        self.b = B
        self.queue = asyncio.Queue()
        self.processed = 0
        self.ts = ts
        self.workers = None
        self.start_time = None
        self.training = False
        self.size = size
        self.ready_workers = [False]*size

        self.arrival_time = []
        self.gradient_time = []
        self.update_time = []
        self.price_distr = price_distr
        self.running = True
        self.cost_log = []

        self.arrival_count = None

        self.net = Net()
        print("param server init")

    def ready_signal(self, worker_index):
        self.ready_workers[worker_index] = True
        #print("READY", self.ready_workers)
        if all(self.ready_workers):
            self.training = True
            print("CALIBRATION COMPLETE: READY TO TRAIN")
            self.ts.ready_signal.remote()
            for w in self.workers:
                w.ready_signal.remote()
        return True

    def signal(self, worker_index, itr):
        #print("got signal from worker {} batch {}".format(worker_index, itr))
        self.queue.put_nowait((worker_index, itr))
        #print(itr)
        return True

    async def queue_consumer(self, workers, test_server, start_time):
        if self.workers is None:
            self.workers = workers
        self.start_time = start_time
        self.arrival_count = np.zeros(len(self.workers))

        # CLEAR QUEUE AFTER CALIBRATION
        while not self.training:
            await asyncio.sleep(0)
        del self.queue
        self.queue = asyncio.Queue()
        #print("QUEUE START")

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
                self.arrival_count[b[0]] += self.b
                self.queue.task_done()
            #print("GROUP COMPLETE")

            group_start = time.time()

            weights = []
            for param in self.net.parameters():
                weights.append(param.data)
            w_ref = ray.put(weights)

            grad = await asyncio.gather(*[self.workers[b[0]].compute_gradients.remote(w_ref, b[1]) for b in batches])

            self.apply_gradients(grad)
            del batches, weights, w_ref, grad
            self.arrival_time.append([self.processed, group_start - self.start_time])
            self.gradient_time.append([self.processed, time.time() - group_start])
            self.update_time.append([self.processed, time.time() - self.start_time])

            if self.processed % self.t == 0:
                print("QUEUE SIZE AT BATCH {} ({}s ELAPSED): {}".format(self.processed, self.update_time[-1][1], self.queue.qsize()))
                self.queue_acc(test_server)

    def apply_gradients(self, gradients):
        #print(type(gradients), [type(g) for g in gradients])
        #print(np.max([np.max([np.max(i) for i in g]) for g in gradients]), np.min([np.min([np.min(i) for i in g]) for g in gradients]))
        #for g in gradients:
            #print(len(g), [(type(i), np.size(i)) for i in g])
            #for i in g:
                #if np.isnan(i).any():
                    #print("AAAAAAAAAAAAAAAA NAN NAN NAN NAN NAN NAN NAN")
                    #print(i)
        #grad = []
        #for i in range(len(gradients[0])):
            #grad.append(np.mean([g[i] for g in gradients]))
        #grad = np.mean(gradients, axis = 0)
        for i, param in enumerate(self.net.parameters()):
            grad = np.mean([g[i] for g in gradients], axis = 0)
            param.data -= self.lr * torch.from_numpy(grad)
        self.processed += self.k
        del grad
        #print(self.processed)

    def queue_acc(self, test_server):
        weights = []
        for param in self.net.parameters():
            weights.append(param.data)

        test_server.test_acc.remote(weights, self.processed)
        del weights

    async def price_producer(self, workers, l, allocation, adaptive=False):
        if self.workers is None:
            self.workers = workers
    
        self.p_spot, update_time = self.price_distr.get_price()
        self.p_on_demand = self.price_distr.get_on_demand()

        while not self.training:
            await asyncio.sleep(0)
        N = len(self.workers)

        #for adaptive method
        self.availability = 1
        self.spot_time = 1
        self.on_time = np.ones(N)

        if adaptive:
            self.persistence = np.ones(N)
        else:
            self.persistence = allocation.allocate(l, self.p_spot, self.p_on_demand)
        prices = self.persistence * self.p_spot
        prices[prices == 0] = self.p_on_demand
        self.spot_state = np.ones(N)
        self.ns = np.sum(self.persistence)
        self.running = np.sum(np.logical_or((1 - self.persistence), self.spot_state))

        last_update = time.time()
        print("starting spot price set to {}".format(self.p_spot))

        refresh_interval = 2
        next_interval = refresh_interval

        total_cost = 0

        while self.running:

            if update_time == False:
                interval = refresh_interval

                self.refresh_workers(allocation, adaptive)
            else:
                if update_time < next_interval:
                    interval = update_time
                    next_interval -= update_time
                    self.p_spot, update_time = self.price_distr.get_price()
                    print("spot price changed to {}".format(self.p_spot))

                    if adaptive:
                        self.adap_allocate(allocation)
                    else:
                        self.persistence = allocation.allocate(l, self.p_spot, self.p_on_demand)
                    prices = self.persistence * self.p_spot
                    prices[prices == 0] = self.p_on_demand
                else:
                    interval = next_interval
                    next_interval = refresh_interval
                    update_time -= interval

                    self.refresh_workers(allocation, adaptive)


            await asyncio.sleep(interval)

            real_interval = time.time() - last_update
            last_update = time.time()
            for i in range(N):
                if not self.persistence[i]:
                    total_cost += real_interval * self.p_on_demand
                    self.on_time[i] += real_interval
                elif self.spot_state[i]:
                    total_cost += real_interval * self.p_spot
                    self.availability += real_interval
                    self.spot_time += real_interval
                    self.on_time[i] += real_interval
                else:
                    self.spot_time += real_interval

            self.cost_log.append([  last_update - self.start_time,
                                    self.p_spot,
                                    np.sum(self.persistence),
                                    np.sum(np.logical_or((1 - self.persistence), self.spot_state)),
                                    total_cost])

        return 'price', np.array(self.cost_log)

    def refresh_workers(self, allocation, adaptive):
        
        switch = allocation.preempt(self.spot_state)
        #if adaptive:
            #self.adap_allocate(allocation)

        self.spot_state = np.logical_xor(self.spot_state, switch).astype(float)
        for i in range(len(self.workers)):
            if self.spot_state[i] or not self.persistence[i]:
                self.workers[i].restart.remote()
            else:
                self.workers[i].preempt.remote()
        new_ns = np.sum(self.persistence)
        new_running = np.sum(np.logical_or((1 - self.persistence), self.spot_state))
        if self.ns != new_ns or self.running != new_running:
            print("NS = {}, number running = {}".format(new_ns, new_running))
            self.ns = new_ns
            self.running = new_running

    def adap_allocate(self, allocation):
        elapsed = time.time() - self.start_time
        l_adap = self.arrival_count / self.on_time
        l_adap[l_adap < 1] = 1
        #print(np.mean(l_adap), self.processed, elapsed, self.availability/self.spot_time)
        a = self.availability/float(self.spot_time)
        self.persistence = allocation.allocate(l_adap, self.p_spot, self.p_on_demand, arrived=self.processed, elapsed=elapsed, a=a)

    def terminate(self):
        self.queue.put_nowait("stop")
        self.running = False

##################################################################
# test server
##################################################################

#@ray.remote(num_cpus=2, num_gpus=1) #GPU MODEL
@ray.remote(num_cpus=4)             #CPU MODEL
class TestServer(object):
    def __init__(self, Net):
        self.processed = 0
        self.weights = {}
        self.queue = asyncio.Queue()
        self.target_itr = -1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_testset_list = False
        self.net = Net().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        self.training = False
        print("test server init on device {}".format(self.device))
    
    def ready_signal(self):
        self.training = True
        return True

    def test_acc(self, weights, itr):
        self.weights[itr] = weights
        self.queue.put_nowait(itr)

    async def valid_consumer(self, get_testset, start_time, expected_itr=200000, target_acc=None, autoexit=False):
        testset = get_testset(self.device.type)
        if self.device.type == 'cpu':
            batch_size = 128
            torch.set_num_threads(4)
        else:
            batch_size = 1024
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        accuracy = []

        while not self.training:
            await asyncio.sleep(0)
        print("TRAINING STARTED, TEST SERVER READY")

        while True:
            itr = await self.queue.get()
            self.queue.task_done()
            if itr == "stop":
                break
            test_start = time.time()

            for i, param in enumerate(self.net.parameters()):
                param.data = self.weights[itr][i].to(self.device)
            del self.weights[itr]

            self.processed = itr
            acc, loss = self.get_acc(test_loader)
            print("ACCURACY AFTER {} BATCHES: {}; LOSS: {}".format(self.processed, acc, loss))
            accuracy.append([self.processed, acc, loss])

            if autoexit and self.target_itr > 0 and self.processed >= max(self.target_itr + 5000, expected_itr):
                print("AUTOEXITING...")
                self.terminate()
                break

            if target_acc and len(accuracy) >= 10 and self.target_itr < 0:
                last_10_acc = np.mean(np.array([a[1] for a in accuracy[-10:]]))
                if last_10_acc > target_acc * 100:
                    self.target_itr = self.processed
                    print("TARGET OF {}% REACHED AFTER {} BATCHES AND {}s AT {}%".format(target_acc * 100, self.target_itr, time.time() - start_time, last_10_acc))

            print("TEST TIME: {}".format(time.time() - test_start))

        return 'ts', np.array(accuracy)

    def get_acc(self, test_loader):
        self.net.eval()
        top1 = data_tools.AverageMeter()
        loss = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            outputs = self.net(inputs.to(self.device))
            l = self.criterion(outputs, targets.to(self.device))
            acc1 = data_tools.comp_accuracy(outputs, targets.to(self.device))
            top1.update(acc1[0], inputs.size(0))
            loss += l.item()
            del outputs, l
        return top1.avg.item(), loss
    
    def terminate(self):
        self.queue.put_nowait("stop")
        return {"target_itr": self.target_itr}

##################################################################
# worker
##################################################################

@ray.remote(num_cpus=2)
class Worker(object):
    def __init__(self, worker_index, ps, Net, B=32, lr=0.03, opt='sgd'):
        self.worker_index = worker_index
        self.ps = ps
        self.curritr = 0
        self.batches = {}
        self.grads = {}
        self.B = B
        self.lr = lr
        self.queue = asyncio.Queue()
        self.training = False
        self.batch_eps = 0
        self.rng = np.random.default_rng()

        self.running = True
        self.preempt = False
        self.arrival_time = []
        self.gradient_time = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net().to(self.device)
        if opt == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=5e-4, betas=(0.9, 0.999), eps=1e-08)
        else:
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accs = []
        print("worker {} init with device {}".format(self.worker_index, self.device))

    def ready_signal(self):
        self.training = True
        return True

    def signal(self, data, target):
        #if torch.isnan(data).any():
            #print(torch.isnan(data).any())
            #print("FOUND NaN; SEED: {}".format(torch.seed()))
            #print(data.numpy())
        self.queue.put_nowait((data, target))
        return True

    async def batch_producer(self, get_trainset, t=0.001):
        trainset = get_trainset(self.worker_index)
        i = 0
        if hasattr(trainset, "is_infinite"):
            infinite_set = True
        else:
            infinite_set = False
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.B, shuffle=True)
            iterator = iter(train_loader)

        self.ps.ready_signal.remote(self.worker_index)

        while not self.training:
            await asyncio.sleep(0)

        start_time = time.time()
        print("WORKER {}: TRAINING READY AT TIME {}".format(self.worker_index, start_time))

        while self.running:
            # SIGNAL QUEUE
            i += 1
            batch_start = time.time()
            if infinite_set:
                data, target = trainset[i]
            else:
                try:
                    data, target = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    data, target = next(iterator)
            self.signal(data, target)
            if infinite_set:
                del data, target

            # SIMULATE DATA INTER-ARRIVAL TIME
            w = self.rng.gamma(self.B, t)
            batch_eps = time.time() - batch_start
            wait_time = np.max([0, w - batch_eps]) #ADJUSTED FOR OTHER DELAYS
            #if self.worker_index == 0:
                #print("RNG: {} BATCH DELAY: {} NEXT BATCH: {}".format(w, batch_eps, wait_time))
            await asyncio.sleep(wait_time)
            #if i == 100 or (i % 1000) == 0:
                #print("ARRIVAL RATE AT {} ARRIVALS: {}".format(i, (time.time() - start_time) / (i * self.B)))

    async def batch_consumer(self, start_time):
        while True:
            data, target = await self.queue.get()
            if data == "stop":
                arrival_time = np.array(self.arrival_time)
                gradient_time = np.array(self.gradient_time)
                return str(self.worker_index), arrival_time, gradient_time
            self.queue.task_done()
            #print(self.curritr, self.preempt)

            if not self.preempt:
                #print("DATA AND TARGET", data[0].size(), target[0])
                data = torch.nan_to_num(data)
                #print(data.shape, target.shape, target.dtype)
                self.batches[self.curritr] = (data, target)
                self.arrival_time.append([self.curritr, time.time() - start_time])

                self.ps.signal.remote(self.worker_index, self.curritr)
                self.curritr += 1

    def compute_gradients(self, weights, itr):
        batch_start = time.time()
        if itr-1 in self.grads:
            del self.grads[itr-1]

        for i, param in enumerate(self.net.parameters()):
            param.data = weights[i].to(self.device)

        data, target = self.batches[itr]
        #with torch.autograd.detect_anomaly(): #CHECK FOR ANOMALY
        self.net.train()
        output = self.net(data.to(self.device))
        #if torch.isnan(output).any():
            #print(output, target)
            #print(torch.isnan(data).any())
        loss = self.criterion(output, target.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()

        self.grads[itr] = []
        for param in self.net.parameters():
            self.grads[itr].append(param.grad.data.numpy())

        del self.batches[itr], data, target, output, loss
        self.gradient_time.append([itr, time.time() - batch_start])
        return self.grads[itr]

    def preempt(self):
        self.preempt = True

    def restart(self):
        self.preempt = False

    def terminate(self):
        self.queue.put_nowait(("stop", None))
        self.running = False;

##################################################################
# coordinator class
##################################################################

class Coordinator(object):
    class Net():
        pass

    def __init__(self, args, pricing):
        self.args = args
        self.ts = TestServer.remote(self.Net)
        self.ps = ParameterServer.remote(self.Net,
                                         self.ts,
                                         pricing,
                                         self.args.size,
                                         k=self.args.K,
                                         t=self.args.test,
                                         B=self.args.bs)
        self.workers = []
        for i in range(self.args.size):
            self.workers.append(Worker.remote(i,
                                              self.ps,
                                              self.Net,
                                              B=self.args.bs,
                                              lr=self.args.lr,
                                              opt=self.args.optimizer))
        self.processes = []

    def run(self):
        raise NotImplementedError

    def autoexit(self):
        raise NotImplementedError

    def terminate(self):
        for w in self.workers:
            w.terminate.remote()
        self.ps.terminate.remote()
        return ray.get(self.ts.terminate.remote())

    def save_logs(self):
        log_list = ray.get(self.processes)
        return log_list
