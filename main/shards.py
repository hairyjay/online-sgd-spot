import numpy as np
import numpy.random as random
import ray

import actors

class Shards(actors.Coordinator):
    def __init__(self, args, pricing):
        super().__init__(args, pricing)

    def testset(self):
        pass

    def trainset(self, idx=None):
        pass

    def get_testset(self, type='cuda'):
        #print(type)
        set = self.testset()
        if type == 'cpu':
            THREADS = 4
            partition_sizes = [1.0 / THREADS for _ in range(THREADS)]
            partition = DataPartitioner(set, partition_sizes, isNonIID=False)
            partitions = [partition.use(i) for i in range(THREADS)]
            return partitions
        else:
            return set

    def get_trainset(self, idx):
        dataset, is_shard = self.trainset(idx)
        if is_shard:
            return dataset
        else:
            partition_sizes = [1.0 / self.args.size for _ in range(self.args.size)]
            partition = DataPartitioner(dataset, partition_sizes, isNonIID=False)
            partition = partition.use(idx)
            return partition

    def run(self, start_time, allocation, rate_dist=None, adaptive=False):
        t = [self.args.t] * self.args.size
        l = 1 / self.args.t
        if rate_dist is not None:
            t, l = rate_dist.get_t(self.args.size)

        self.processes.append(self.ps.queue_consumer.remote(self.workers, self.ts, start_time))
        self.processes.append(self.ps.price_producer.remote(self.workers, l, allocation, self.args.adap))
        self.processes.append(self.ts.valid_consumer.remote(self.get_testset,
                                                            start_time,
                                                            expected_itr=self.args.J,
                                                            target_acc=self.args.target,
                                                            autoexit=self.args.autoexit))

        for i, w in enumerate(self.workers):
            self.processes.extend([w.batch_producer.remote(self.get_trainset, t=t[i]), w.batch_consumer.remote(start_time)])

    def autoexit(self):
        if self.args.autoexit:
            log_list = [ray.get(self.processes[2])]
            self.processes.pop(2)
            test_stats = self.terminate()
            print("TERMINATED")
            log_list.extend(self.save_logs())
            return log_list, test_stats
        else:
            return False

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chunks. """
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234, isNonIID=False):
        self.data = data
        self.partitions = []
        rng = random.default_rng(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)


        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

        if isNonIID:
            self.partitions = __getNonIIDdata__(self, data, sizes, seed)

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
