import numpy as np
import ray
import asyncio
import time

import torch
import torch.optim as optim
#import torch.distributed as dist
#import torch.multiprocessing as mp

from . import data_tools
from . import shards
#from . import price

##################################################################
# parameter server
##################################################################

@ray.remote(num_cpus=4)
class ParameterServer(object):
    def __init__(self, classes, Net, ts, pr, size, time_scale, lr=0.005, k=5, t=100, B=256):
        self.params = 0
        self.lr = lr
        self.k = k
        self.t = t
        self.b = B
        self.queue = asyncio.Queue()
        self.processed = 0
        self.ts = ts
        self.pr = pr
        self.workers = None
        self.start_time = None
        self.training = False
        self.size = size
        self.ready_workers = [False]*size

        self.arrival_time = []
        self.gradient_time = []
        self.update_time = []
        self.running = True
        self.time_scale = time_scale

        self.arrival_count = None

        self.net = Net(classes)
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

    async def queue_consumer(self, workers, start_time):
        if self.workers is None:
            self.workers = workers
        self.start_time = start_time
        self.arrival_count = np.zeros(len(self.workers))

        # CLEAR QUEUE AFTER CALIBRATION
        while not self.training:
            await asyncio.sleep(0)
        del self.queue
        self.queue = asyncio.Queue()
        print("QUEUE START")

        while True:
            batches = []
            for i in range(self.k):
                b = await self.queue.get()
                if b == "stop":
                    arrival_time = np.array(self.arrival_time) / self.time_scale
                    gradient_time = np.array(self.gradient_time) / self.time_scale
                    update_time = np.array(self.update_time) / self.time_scale
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
            self.pr.count_signal.remote(self.arrival_count, self.processed, group_start - self.start_time)

            if self.processed % self.t == 0:
                print("QUEUE SIZE AT BATCH {}, {:.0f}s: {}".format(self.processed, self.update_time[-1][1], self.queue.qsize()))
                self.queue_acc()
            
            await asyncio.sleep(0)

    def apply_gradients(self, gradients):
        for i, param in enumerate(self.net.parameters()):
            grad = np.mean([g[i] for g in gradients], axis = 0)
            param.data -= self.lr * torch.from_numpy(grad)
        self.processed += self.k
        del grad

    def queue_acc(self):
        weights = []
        for param in self.net.parameters():
            weights.append(param.data)

        self.ts.test_acc.remote(weights, self.processed)
        del weights

    def terminate(self):
        self.queue.put_nowait("stop")
        self.running = False

##################################################################
# price server
##################################################################

@ray.remote(num_cpus=2)
class PriceServer(object):
    def __init__(self, price_distr, time_scale):
        self.price_distr = price_distr
        self.workers = None
        self.start_time = None
        self.cost_log = []
        self.arrival_count = None
        self.processed = 0
        self.ps_time = 1
        self.time_scale = time_scale

    def count_signal(self, arrival_count, processed, time):
        self.arrival_count = arrival_count
        self.processed = processed
        self.ps_time = time
        return True

    async def price_producer(self, workers, start_time, l, allocation, adaptive=False):
        if self.workers is None:
            self.workers = workers
        self.arrival_count = np.zeros(len(self.workers))
    
        self.p_spot, update_time = self.price_distr.get_price()
        self.p_on_demand = self.price_distr.get_on_demand()
    
        N = len(self.workers)
        self.start_time = start_time

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

        refresh_interval = 2 * self.time_scale
        next_interval = refresh_interval
        last_refresh = time.time()

        total_cost = 0

        while self.running:

            if update_time == False:
                interval = refresh_interval

                self.refresh_workers(allocation, adaptive, last_refresh)
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

                    self.refresh_workers(allocation, adaptive, last_refresh)

            last_refresh = time.time()

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

            # PRICE LOG OUTPUT
            #   0: TIMESTAMP AT UPDATE TIME
            #   1: SPOT PRICE
            #   2: NUMBER OF SPOT INSTANCES
            #   3: NUMBER OF ONLINE INSTANCES
            #   4: REAL RECORDED COST
            #print("time: {}, processed: {}, total cost: {}".format((last_update - self.start_time) / self.time_scale, self.processed, total_cost))
            self.cost_log.append([  (last_update - self.start_time) / self.time_scale,
                                    self.p_spot,
                                    np.sum(self.persistence),
                                    np.sum(np.logical_or((1 - self.persistence), self.spot_state)),
                                    total_cost])

        return 'price', np.array(self.cost_log)

    def refresh_workers(self, allocation, adaptive, interval):
        
        switch = allocation.preempt(self.spot_state)
        if adaptive:
            self.adap_allocate(allocation)

        self.spot_state = np.logical_xor(self.spot_state, switch).astype(float)
        for i in range(len(self.workers)):
            if self.spot_state[i] or not self.persistence[i]:
                self.workers[i].restart.remote()
            else:
                self.workers[i].preempt.remote()
        new_ns = np.sum(self.persistence)
        new_running = np.sum(np.logical_or((1 - self.persistence), self.spot_state))
        if self.ns != new_ns or self.running != new_running:
            print("NS = {}, number running = {}, since last refresh = {}, p_spot = {}, p_od = {}".format(new_ns, new_running, time.time() - interval, self.p_spot, self.p_on_demand))
            self.ns = new_ns
            self.running = new_running

    def adap_allocate(self, allocation):
        l_adap = self.arrival_count / self.on_time
        l_adap[l_adap < 1] = 1
        #print(np.mean(l_adap), self.processed, elapsed, self.availability/self.spot_time)
        a = self.availability/float(self.spot_time)
        self.persistence = allocation.allocate(l_adap, self.p_spot, self.p_on_demand, arrived=self.processed, elapsed=self.ps_time, a=a)

    def terminate(self):
        self.running = False

##################################################################
# test server
##################################################################

@ray.remote(num_cpus=4, num_gpus=1) #GPU MODEL
#@ray.remote(num_cpus=4)             #CPU MODEL
class TestServer(object):
    def __init__(self, Net, classes, drift={}):
        self.processed = 0
        self.weights = {}
        self.queue = asyncio.Queue()
        self.target_itr = -1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_testset_list = False
        self.net = Net(classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        self.drift = drift
        self.drift_weights = None
        self.drift_map = np.arange(classes, dtype=np.int32)

        self.sampler = None
        self.loader_len = 0
        self.training = False
        print("test server init on device {}".format(self.device))
    
    def ready_signal(self):
        self.training = True
        return True

    def test_acc(self, weights, itr):
        self.weights[itr] = weights
        self.queue.put_nowait(itr)

    async def valid_consumer(self, get_testset, get_augment, start_time, expected_itr=1000, target_acc=None, autoexit=False, mask=None):
        testset = get_testset(self.device.type)
        self.augment = get_augment()
        torch.set_num_threads(4)
        if self.device.type == 'cpu':
            batch_size = 128
        else:
            batch_size = 4096

        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        self.loader_len = len(test_loader)
        #print(self.loader_len)
        if self.drift:
            #self.drift_weights = np.ones(len(mask[1]))
            self.drift_weights = np.ones(testset.targets.size())
            self.drift_withdrawn = len(mask[0]) // 2
            self.drift_mask_p = np.concatenate([np.argwhere(testset.targets == c) for c in mask[0][:self.drift_withdrawn]], axis=None)
            self.drift_mask_n = np.concatenate([np.argwhere(testset.targets == c) for c in mask[0][self.drift_withdrawn:]], axis=None)
            self.drift_weights[self.drift_mask_n] = 0
            #print(self.drift_mask, len(self.drift_mask), self.drift_withdrawn, self.drift_weights)
            #print(testset.targets.size())
            self.sampler = torch.utils.data.WeightedRandomSampler(self.drift_weights, batch_size)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=self.sampler)
            self.drift_map = torch.from_numpy(mask[1]).long()
        else:
            self.drift_map = torch.from_numpy(self.drift_map).long()

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
            test_time = test_start - start_time
            print(test_time)
            if self.drift:
                if test_time > self.drift["start"]:
                    if test_time > self.drift["start"] + self.drift["time"]:
                        self.drift_weights[self.drift_mask_p] = 0
                        self.drift_weights[self.drift_mask_n] = 1
                    else:
                        self.drift_weights[self.drift_mask_p] = 1 - (test_time - self.drift["start"]) / self.drift["time"]
                        self.drift_weights[self.drift_mask_n] = (test_time - self.drift["start"]) / self.drift["time"]
                    self.sampler.weights = torch.as_tensor(self.drift_weights)
            for i, param in enumerate(self.net.parameters()):
                param.data = self.weights[itr][i].to(self.device)
            del self.weights[itr]

            self.processed = itr
            acc, loss, count = self.get_acc(test_loader)
            print("AFTER {} BATCHES: {:.2f}% ACC; {:.0f} LOSS; {:.0f} COUNT".format(self.processed, acc, loss, count))
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
        def compute_acc(inputs, targets, top1):
            inputs = self.augment(inputs.to(self.device))
            targets = targets.to(self.device)
            outputs = self.net(inputs)
            l = self.criterion(outputs, targets)
            acc1 = data_tools.comp_accuracy(outputs, targets)
            top1.update(acc1[0], inputs.size(0))
            del outputs
            #a = time.time()
            return l.item()
        self.net.eval()
        top1 = data_tools.AverageMeter()
        loss = 0
        if self.drift:
            count = np.zeros(len(self.drift_map))
            iterator = iter(test_loader)
            for i in range(self.loader_len):
                try:
                    inputs, targets = next(iterator)
                except StopIteration:
                    iterator = iter(test_loader)
                    inputs, targets = next(iterator)
                #print(targets)
                #print(self.drift_mask)
                u, c = np.unique(targets, return_counts=True)
                #print(u, c)
                count[u] += c
                loss += compute_acc(inputs, self.drift_map[targets], top1)
            #print(self.drift_weights)
            #print(self.drift_map)
            #print(count)
        else:
            for i, (inputs, targets) in enumerate(test_loader):
                loss += compute_acc(inputs, self.drift_map[targets], top1)
        return top1.avg.item(), loss, top1.count
    
    def terminate(self):
        self.queue.put_nowait("stop")
        return {"target_itr": self.target_itr}

##################################################################
# worker
##################################################################

@ray.remote(num_cpus=2)
class Worker(object):
    def __init__(self, worker_index, ps, classes, Net, time_scale, B=32, lr=0.03, opt='sgd', drift={}):
        self.worker_index = worker_index
        self.ps = ps
        self.curritr = 0
        self.batches = {}
        self.B = B
        self.lr = lr
        self.queue = asyncio.Queue()
        self.training = False
        self.batch_eps = 0
        self.rng = np.random.default_rng()
        self.drift = drift
        self.drift_weights = None
        self.sampler = None
        self.drift_mask = None
        self.drift_map = np.arange(classes, dtype=np.int32)

        self.running = True
        self.preempt = False
        self.arrival_time = []
        self.gradient_time = []
        self.time_scale = time_scale
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(classes).to(self.device)
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

    def signal(self, signal):
        self.queue.put_nowait(signal)
        return True

    async def batch_producer(self, get_trainset, get_augment, t=0.001, mask=None):
        #print(self.worker_index, 1/t, t)
        torch.set_num_threads(4)
        trainset = get_trainset(self.worker_index)
        self.augment = get_augment()
        i = 0
        # ADD SAMPLER HERE
        if self.drift:
            try:
                if isinstance(trainset, shards.Partition):
                    targets = trainset.data.targets[trainset.index]
                else:
                    targets = trainset.targets
                self.drift_weights = np.ones(targets.size())
                self.drift_withdrawn = len(mask[0]) // 2
                self.drift_mask_p = np.concatenate([np.argwhere(targets == c) for c in mask[0][:self.drift_withdrawn]], axis=None)
                self.drift_mask_n = np.concatenate([np.argwhere(targets == c) for c in mask[0][self.drift_withdrawn:]], axis=None)
                self.drift_weights[self.drift_mask_n] = 0
                self.sampler = torch.utils.data.WeightedRandomSampler(self.drift_weights, self.B)
                self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.B, sampler=self.sampler)
                self.drift_map = torch.from_numpy(mask[1]).long()
                #print(self.drift_weights.size, self.drift_mask_p.size, self.drift_mask_n.size)
            except:
                import traceback
                traceback.print_exc()
        else:
            self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.B, shuffle=True)
            self.drift_map = torch.from_numpy(self.drift_map).long()
        self.iterator = iter(self.train_loader)

        self.ps.ready_signal.remote(self.worker_index)

        while not self.training:
            await asyncio.sleep(0)

        start_time = time.time()

        while self.running:
            # SIGNAL QUEUE
            i += 1
            batch_start = time.time()
            batch_time = batch_start - start_time
            if self.drift:
                if batch_time > self.drift["start"]:
                    if batch_time > self.drift["start"] + self.drift["time"]:
                        self.drift_weights[self.drift_mask_p] = 0
                        self.drift_weights[self.drift_mask_n] = 1
                    else:
                        self.drift_weights[self.drift_mask_p] = 1 - (batch_time - self.drift["start"]) / self.drift["time"]
                        self.drift_weights[self.drift_mask_n] = (batch_time - self.drift["start"]) / self.drift["time"]
                try:
                    self.sampler.weights = torch.as_tensor(self.drift_weights)
                except:
                    print("ya fucked it")
            self.signal(i)

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
            signal = await self.queue.get()
            if signal == "stop":
                arrival_time = np.array(self.arrival_time) / self.time_scale
                gradient_time = np.array(self.gradient_time) / self.time_scale
                return str(self.worker_index), arrival_time, gradient_time
            self.queue.task_done()
            #print(self.curritr, self.preempt)

            if not self.preempt:
                self.batches[self.curritr] = signal
                self.arrival_time.append([self.curritr, time.time() - start_time])

                self.ps.signal.remote(self.worker_index, self.curritr)
                self.curritr += 1

    def compute_gradients(self, weights, itr):
        batch_start = time.time()

        for i, param in enumerate(self.net.parameters()):
            param.data = weights[i].to(self.device)

        try:
            data, target = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.train_loader)
            data, target = next(self.iterator)
        #print(self.worker_index, target, self.drift_map)
        map_target = self.drift_map[target]
        #with torch.autograd.detect_anomaly(): #CHECK FOR ANOMALY
        self.net.train()
        aug_data = self.augment(data.to(self.device))
        output = self.net(aug_data)
        #print(self.worker_index, output.shape)
        #if torch.isnan(output).any():
            #print(output, target)
            #print(torch.isnan(data).any())
        map_target.to(self.device)
        #print("target to device")
        loss = self.criterion(output, map_target)
        #print(self.worker_index, "loss:", loss.shape)
        self.optimizer.zero_grad()
        loss.backward()

        grads = []
        for param in self.net.parameters():
            grads.append(param.grad.data.numpy())

        del self.batches[itr], data, aug_data, target, map_target, output, loss
        self.gradient_time.append([itr, time.time() - batch_start])
        return grads

    def preempt(self):
        self.preempt = True

    def restart(self):
        self.preempt = False

    def terminate(self):
        self.queue.put_nowait("stop")
        self.running = False;

##################################################################
# coordinator class
##################################################################

class Coordinator(object):
    class Net():
        pass

    def __init__(self, args, pricing, drift):
        self.args = args
        self.ts = TestServer.remote(self.Net, self.classes, drift=drift)
        self.pr = PriceServer.remote(pricing, self.args.time_scale)
        self.ps = ParameterServer.remote(self.classes,
                                         self.Net,
                                         self.ts,
                                         self.pr,
                                         self.args.size,
                                         self.args.time_scale,
                                         k=self.args.K,
                                         t=self.args.test,
                                         B=self.args.bs)
        self.workers = []
        for i in range(self.args.size):
            self.workers.append(Worker.remote(i,
                                              self.ps,
                                              self.classes,
                                              self.Net,
                                              self.args.time_scale,
                                              B=self.args.bs,
                                              lr=self.args.lr,
                                              opt=self.args.optimizer,
                                              drift=drift))
        self.processes = []

    def run(self):
        raise NotImplementedError

    def autoexit(self):
        raise NotImplementedError

    def terminate(self):
        for w in self.workers:
            w.terminate.remote()
        self.ps.terminate.remote()
        self.pr.terminate.remote()
        return ray.get(self.ts.terminate.remote())

    def save_logs(self):
        log_list = ray.get(self.processes)
        return log_list
