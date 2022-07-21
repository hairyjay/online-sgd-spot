import numpy as np
import numpy.random as random
import ray
import asyncio
import time

##################################################################
# Start ray cluster
##################################################################

#ray.init(redis_address="127.0.0.1:6379")
ray.init(address="auto")
#ray.init(local_mode=True)

##################################################################
# parameter server
##################################################################

@ray.remote(num_cpus=4)
class ParameterServer(object):
    def __init__(self, k=5):
        self.params = 0
        self.k = k
        self.queue = asyncio.Queue()
        print("server init")

    def signal(self, worker_index, itr):
        print("got signal from worker {} batch {}".format(worker_index, itr))
        self.queue.put_nowait((worker_index, itr))
        return True

    async def queue_processor(self, workers):
        while True:
            batches = []
            for i in range(self.k):
                b = await self.queue.get()
                batches.append(b)

            print("running gradient computation")
            grad = await asyncio.gather(*[workers[b[0]].compute_gradients.remote(b[1]) for b in batches])

            self.apply_gradients(grad)

    def apply_gradients(self, gradients):
        for g in gradients:
            self.params -= g
        print(self.params)

##################################################################
# worker
##################################################################

@ray.remote(num_cpus=4)
class Worker(object):
    def __init__(self, worker_index, ps, B=1024, l=0.005):
        self.worker_index = worker_index
        self.ps = ps
        self.curritr = 0
        self.batches = {}
        self.B = B
        self.l = l
        print("worker {} init".format(self.worker_index))

    async def batch_generator(self):
        while True:
            # SIMULATE MINI-BATCH INTER-ARRIVAL TIME
            wait_time = random.gamma(self.B, self.l)
            #print("wait time: {}".format(wait_time))
            await asyncio.sleep(wait_time)
            print("WORKER {} BATCH {} ARRIVED".format(self.worker_index, self.curritr))

            # SIGNAL PARAMETER SERVER
            self.batches[self.curritr] = [0]
            self.ps.signal.remote(self.worker_index, self.curritr)
            self.curritr += 1

    async def compute_gradients(self, itr):
        grad = -1
        del self.batches[itr]
        print("worker {} gradient of batch {} computed".format(self.worker_index, itr))
        return grad

if __name__ == "__main__":
    num_workers = 8

    # Initialize workers and parameter server
    ps = ParameterServer.remote()
    workers = []
    for i in range(num_workers):
        workers.append(Worker.remote(i, ps))
    print("servers launched")

    processes = []
    processes.append(ps.queue_processor.remote(workers))
    for w in workers:
        processes.append(w.batch_generator.remote())
    print("processes launched")

    print("waiting for the end of time")
    ray.wait(processes)
