import numpy as np
import ray

from . import actors

class Shards(actors.Coordinator):
    def __init__(self, args, pricing, drift, classes=0):
        self.drift_mask = (np.empty(shape=0), np.empty(shape=0), None)
        if drift:
            if sum(drift["cats"])*2 >= classes:
                raise ValueError("Withheld classes for drift may not exceed half of total classes")
            if drift["rand"]:
                self.drift_classes = np.random.choice(classes, size=sum(drift["cats"])*2, replace=False)
            else:
                self.drift_classes = np.arange(classes-sum(drift["cats"])*2, classes)
            print(self.drift_classes)
            self.drift_map = np.zeros(classes, dtype=np.int32)
            self.drift_map -= 1
            self.drift_map[self.drift_classes[:sum(drift["cats"])]] = np.arange(sum(drift["cats"]))
            self.drift_map[self.drift_classes[sum(drift["cats"]):]] = np.arange(sum(drift["cats"]))
            i = sum(drift["cats"])
            for n in range(len(self.drift_map)):
                if self.drift_map[n] == -1:
                    self.drift_map[n] = i
                    i += 1
            self.drift_mask = (self.drift_classes, self.drift_map, classes - sum(drift["cats"]))
            self.classes = classes - sum(drift["cats"])
        super().__init__(args, pricing, drift)

    def testset(self):
        pass

    def trainset(self, idx=None):
        pass

    def get_testset(self, type='cuda'):
        return self.testset()

    def get_trainset(self, idx):
        dataset, is_shard = self.trainset(idx)
        if is_shard:
            return dataset
        else:
            partition_sizes = [1.0 / self.args.size for _ in range(self.args.size)]
            partition = DataPartitioner(dataset, partition_sizes, isNonIID=False)
            partition = partition.use(idx)
            return partition
        
    def get_test_augment(self):
        pass
        
    def get_train_augment(self):
        pass

    def run(self, start_time, allocation, rate_dist=None, adaptive=False):
        t = [self.args.t] * self.args.size
        l = 1 / self.args.t
        if rate_dist is not None:
            t, l = rate_dist.get_t(self.args.size)

        self.processes.append(self.ps.queue_consumer.remote(self.workers, start_time))
        self.processes.append(self.pr.price_producer.remote(self.workers, start_time, l, allocation, self.args.adap))
        self.processes.append(self.ts.valid_consumer.remote(self.get_testset,
                                                            self.get_test_augment,
                                                            start_time,
                                                            expected_itr=self.args.J,
                                                            target_acc=self.args.target,
                                                            autoexit=self.args.autoexit,
                                                            force_exit=self.args.force_exit,
                                                            mask=self.drift_mask))
        print("ready to start batch_producer tasks")

        for i, w in enumerate(self.workers):
            self.processes.extend([w.batch_producer.remote(self.get_trainset,
                                                           self.get_train_augment,
                                                           t=t[i],
                                                           mask=self.drift_mask),
                                   w.batch_consumer.remote(start_time)])

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
        rng = np.random.default_rng(seed)
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
    
def drift_split(list, lens):
    n = 0
    split_list = []
    for i in lens:
        split_list.append(list[n:n+i])
        n += i
    return split_list