import numpy as np
import ray
import time
import argparse
import os
from math import ceil

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import random
from random import Random

from pathlib import Path

##################################################################
# MNIST model
##################################################################

class MNIST_Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

##################################################################
# CIFAR-10 model
##################################################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

##################################################################
# Utils
##################################################################

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
    """ Partitions a dataset into different chuncks. """
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234, isNonIID=False):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
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

def partition_dataset(rank, size, bsz):
    print('==> load train data')
    # transform=transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.1307,), (0.3081,))
    # ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='~/spot_aws/data',
                                            train=True,
                                            download=True,
                                            transform=transform_train)

    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(trainset, partition_sizes, isNonIID=False)
    partition = partition.use(rank)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='~/spot_aws/data',
                                        train=False,
                                        download=True,
                                        transform=transform_test)

    return partition, testset

def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

##################################################################
# Start ray cluster
##################################################################

ray.init(address="auto")

##################################################################
# parameter server
##################################################################

@ray.remote(num_cpus=4)
class ParameterServer(object):
    def __init__(self, testset, learning_rate=0.1):
        self.net = Net()
        self.lr = learning_rate
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.test_loader = torch.utils.data.DataLoader(testset,
                                            batch_size=128,
                                            shuffle=False)
        print("server init!!")

    def apply_gradients(self, *gradients):
        grads = np.mean(gradients, axis = 0)
        i = 0
        for param in self.net.parameters():
            param.data -= self.lr * torch.Tensor(grads[i])
            i += 1

        params = []
        for param in self.net.parameters():
            params.append(param.data)

        return params

    def get_weights(self):
        params = []
        for param in self.net.parameters():
            params.append(param.data)

        return params

    def test_acc(self):
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

##################################################################
# worker
##################################################################

@ray.remote(num_cpus=4)
class Worker(object):
    def __init__(self, worker_index, trainset, batch_size=128, size = 4, curritr=0):
        self.worker_index = worker_index
        self.batch_size = batch_size


        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
        self.iterator = iter(self.train_loader)
        print(len(self.train_loader))
        self.net = Net()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.1, momentum=0, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.trainacc = AverageMeter()
        self.accs = []

        self.curritr = curritr
        # self.currGradient = currGradient

    def compute_gradients(self, weights):
        i = 0
        for param in self.net.parameters():
            param.data = weights[i]
            i += 1

        try:
            data, target = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.train_loader)
            data, target = next(self.iterator)

        # data, target = data.cuda(), target.cuda()
        output = self.net(data)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.curritr += 1

        grads = []
        for param in self.net.parameters():
            grads.append(param.grad.data.numpy())

        return grads

    def getWorkerIndex(self):
        return self.worker_index

    def getworkerAccs(self):
        return self.accs

# Three different strategies:
# Static K = num_workers:
#                   lr = 0.05
#
#                   saveFoldername: ... 'static-n'
# Dynamic K = 1 * eta ** :
#                   lr = 0.05
#                   b1, b2 = 1, 1
#                   saveFoldername: ... 'non-interruptible'
# Optimal single bid:
#                   lr = 0.05
#                   b1, b2 = 0.07069, 0.07069
#                   saveFoldername: ... 'single'

parser = argparse.ArgumentParser(description='PyTorch K-sync SGD')
parser.add_argument('--name','-n', default="default", type=str, help='experiment name, used for saving results')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
#parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=32, type=int, help='batch size on each worker')
parser.add_argument('--epoch', '-e', default=100, type=int, help='total epoch')
parser.add_argument('--size', default=8, type=int, help='number of workers')
parser.add_argument('--K', default=8, type=int, help='number of returned workers')
parser.add_argument('--save', '-s', default=True, action='store_true', help='whether save the training results')
args = parser.parse_args()

if __name__ == "__main__":

    num_workers = args.size
    K = args.K
    print(args.K)

    # Initialize workers and parameter server
    workers = []
    for i in range(num_workers):
        trainset, testset = partition_dataset(i, num_workers, args.bs)
        workers.append(Worker.remote(i, trainset, args.bs, args.size))
    ps = ParameterServer.remote(testset, learning_rate = args.lr)

    # Initialize available worker list
    available_workers = [i for i in range(num_workers)]

    # Auxiliary variables for logging
    num_batches = ceil(len(trainset)/float(args.bs))
    epoch_time = 0
    cost = 0
    record_time = []
    record_acc = []
    record_cost = []


    # Calculate the number of active workers based on price data and bid prices
    instance = 'c5.xlarge'
    select_region = 'us-west-2a'

    current_price = 0.1

    # Start training
    current_weights = ray.get(ps.get_weights.remote())

    total_iters = num_batches*args.epoch
    # Static strategy 1: K = 1

    # total_iters = num_batches*arg.epoch / 2
    # Static strategy 2: K = 2,

    total_iters = num_batches * 26
    # Dynamic strategy: total_iters = log(total_iters(eta -1)) / log(eta**chi), chi = 1
    #                               = 196*26
    for iters in range(total_iters):


        # Static strategy 1: K = 1
        # K = 1

        # Static strategy 2: K = 2

        # Dynamic strategy:
        eta = 1.0004 #(maximum K will be smaller than 8 for 10000 iterations)
        8 > eta ** (np.log(10000*(eta - 1))/np.log(eta))
        K = np.ceil(1 * eta ** iters)

        tic = time.time()
        # fobj_to_workerID_dict = {}
        task_list = []
        for worker_id in available_workers:
            worker = workers[worker_id]
            remotefn = worker.compute_gradients.remote(current_weights)
            task_list.append(remotefn)
            # fobj_to_workerID_dict[remotefn] = worker_id

        random.shuffle(task_list)
        fast_function_ids, straggler_function_ids  = ray.wait(task_list, num_returns=K, timeout=10.0)
        fast_gradients = [ray.get(fast_id) for fast_id in fast_function_ids]
        current_weights = ray.get(ps.apply_gradients.remote(*fast_gradients))




        iter_time = time.time() - tic
        epoch_time += iter_time



        ray.wait(task_list, num_returns=num_workers)
        # fast_worker_IDs = [fobj_to_workerID_dict[fast_id] for fast_id in fast_function_ids]

        # Compute test accuracy of the central model
        # Compute cost
        if iters % num_batches == 0:
            testacc = ray.get(ps.test_acc.remote())
            iter_cost = current_price * K / 3600 * iter_time
            cost += iter_cost
            record_acc.append(testacc)
            record_time.append(epoch_time)
            record_cost.append(cost)

            print('iteration: %d, time: %.3f, acc: %.3f, price: %.5f, K: %.1f, iter_cost: %.9f' % (iters, epoch_time, \
                testacc, current_price, K, iter_cost))
            # epoch_time = 0

    save_path = './results/'

    #saveFolderName = save_path + str(int(time.time())) + '_' + 'K1'+ '_' + args.name + '_' + str(instance)+'_' + 'K' + str(args.K) + '_' + 'P' + str(num_workers) + '_lr'+str(args.lr)

    saveFolderName = save_path + str(int(time.time())) + '_' + 'dynamic' + '_' + args.name + '_' + str(instance)+'_' + 'K' + str(args.K) + '_' + 'P' + str(num_workers) + '_lr'+str(args.lr)

    # saveFolderName = save_path + str(int(time.time())) + '_' + 'K2' + '_' + args.name + '_' + str(instance)+'_' + 'K' + str(args.K) + '_' + 'P' + str(num_workers) + '_lr'+str(args.lr)

    print('saveFolderName: {}'.format(saveFolderName))
    if os.path.isdir(saveFolderName)==False and args.save:
        os.mkdir(saveFolderName)
        np.savetxt(saveFolderName+'/timing.log', record_time, delimiter=',')
        np.savetxt(saveFolderName+'/accuracy.log', record_acc, delimiter=',')
        np.savetxt(saveFolderName+'/cost.log', record_cost, delimiter=',')
