import numpy as np
import numpy.random as random
import ray
import asyncio
import time

import torch
import torch.optim as optim

import data_tools

##################################################################
# parameter server
##################################################################

@ray.remote(num_cpus=4)
class ParameterServer(object):
    def __init__(self, Net, price_distr, l=0.005, k=5, t=100):
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

    async def queue_consumer(self, workers, test_server, start_time):
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

            grad = await asyncio.gather(*[self.workers[b[0]].compute_gradients.remote(w_ref, b[1]) for b in batches])

            self.apply_gradients(grad)
            del batches, weights, w_ref, grad#, tasks
            self.arrival_time.append([self.processed, group_start - self.start_time])
            self.gradient_time.append([self.processed, time.time() - group_start])
            self.update_time.append([self.processed, time.time() - self.start_time])

            if self.processed % self.t == 0:
                print("QUEUE SIZE: {}".format(self.queue.qsize()))
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

    async def price_producer(self, bid1, bid2=None, n2=0):
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
            if interval == False:
                break
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
    def __init__(self, Net):
        self.processed = 0
        self.weights = {}
        self.queue = asyncio.Queue()
        self.target_itr = -1
        self.net = Net()
        print("test server init")

    def test_acc(self, weights, itr):
        self.weights[itr] = weights
        self.queue.put_nowait(itr)

    async def valid_consumer(self, get_testset, start_time, target_acc=None):
        testset = get_testset()
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        accuracy = []

        while True:
            itr = await self.queue.get()
            self.queue.task_done()
            if itr == "stop":
                return 'ts', np.array(accuracy)

            for i, param in enumerate(self.net.parameters()):
                param.data = self.weights[itr][i]
            del self.weights[itr]

            self.processed = itr
            acc = self.get_acc(test_loader)
            print("ACCURACY AFTER {} BATCHES: {}".format(self.processed, acc))
            accuracy.append([self.processed, acc])

            if target_acc and len(accuracy) >= 10 and self.target_itr < 0:
                last_10_acc = np.mean(np.array([a[1] for a in accuracy[-10:]]))
                if last_10_acc > target_acc * 100:
                    self.target_itr = self.processed
                    print("TARGET OF {}% REACHED AFTER {} BATCHES AND {}s AT {}%".format(target_acc * 100, self.target_itr, time.time() - start_time, last_10_acc))

    def get_acc(self, test_loader):
        self.net.eval()
        top1 = data_tools.AverageMeter()
        # correct = 0
        # total = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = self.net(inputs)
            acc1 = data_tools.comp_accuracy(outputs, targets)
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
    def __init__(self, worker_index, ps, Net, B=32, lr=0.03):
        self.worker_index = worker_index
        self.ps = ps
        self.curritr = 0
        self.batches = {}
        self.B = B
        self.lr = lr
        self.queue = asyncio.Queue()

        self.running = True
        self.preempt = False
        self.arrival_time = []
        self.gradient_time = []
        self.net = Net()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accs = []
        print("worker {} init".format(self.worker_index))

    async def batch_producer(self, get_trainset, t=0.001):
        trainset = get_trainset(self.worker_index)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
        iterator = iter(train_loader)

        while self.running:
            # SIGNAL QUEUE
            try:
                data, target = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                data, target = next(iterator)
            self.signal(data, target)

            # SIMULATE DATA INTER-ARRIVAL TIME
            wait_time = random.exponential(t)
            await asyncio.sleep(wait_time)

    def signal(self, data, target):
        self.queue.put_nowait((data, target))
        return True

    async def batch_consumer(self, start_time):
        while True:
            data, target = [], []
            for i in range(self.B):
                d, t = await self.queue.get()
                if d == "stop":
                    arrival_time = np.array(self.arrival_time)
                    gradient_time = np.array(self.gradient_time)
                    return str(self.worker_index), arrival_time, gradient_time
                data.append(d)
                target.append(t)
                self.queue.task_done()

            if not self.preempt:
                data = torch.cat(data, 0)
                target = torch.cat(target, 0)
                self.batches[self.curritr] = (data, target)
                self.arrival_time.append([self.curritr, time.time() - start_time])

                self.ps.signal.remote(self.worker_index, self.curritr)
                self.curritr += 1

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
        return grads

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
        self.ps = ParameterServer.remote(self.Net, pricing, k=self.args.K, t=self.args.test)
        self.ts = TestServer.remote(self.Net)
        self.workers = []
        for i in range(self.args.size):
            self.workers.append(Worker.remote(i, self.ps, self.Net, B=self.args.bs, lr=self.args.lr))
        self.processes = []

    def run(self):
        pass

    def terminate(self):
        for w in self.workers:
            w.terminate.remote()
        self.ps.terminate.remote()
        return ray.get(self.ts.terminate.remote())

    def save_logs(self):
        log_list = ray.get(self.processes)
        return log_list
