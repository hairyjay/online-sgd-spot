
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch import linalg
from torch.amp import autocast

from torchvision import datasets, transforms
import torchvision.models as models

import time

class Params:
    def __init__(self):
        self.device = torch.device('cuda')
        self.world_size = torch.cuda.device_count()
        self.batch_size = 256 * self.world_size
        self.workers = 40
        self.max_lr = 0.175
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.epochs = 50
        self.pct_start = 0.3
        self.div_factor = 25.0
        self.final_div_factor = 1e4
        # self.lr = 0.1
        # self.momentum = 0.9
        # self.weight_decay = 1e-4
        # self.lr_step_size = 30
        # self.lr_gamma = 0.1

class Net(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, classes)

    def forward(self, x):
        return self.model(x)

def train(params, train_loader, test_loader, model, loss_fn, optimizer, epoch):
    model.train()
    start_time = time.time()
    torch.set_num_threads(params.workers)
    print("workers: {}, devices: {}".format(params.workers, params.world_size))

    for batch_idx, data_batch in enumerate(train_loader):
        # batch_start_time = time.time()
        inputs, labels = data_batch
        inputs, labels = inputs.to(params.device), labels.to(params.device)
        # print("{}A: {:.2f}s".format(batch_idx, time.time()-batch_start_time))
        chkpt = time.time()

        # Compute prediction error
        with autocast(device_type=params.device.type):
            pred = model(inputs)
            loss = loss_fn(pred, labels)
        # print("{}B: {:.2f}s".format(batch_idx, time.time()-chkpt))
        # chkpt = time.time()

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch_idx % 250 == 125:
            for i, param in enumerate(model.parameters()):
                print("Epoch {} Batch {} Layer {}: Norm = {}".format(epoch, batch_idx, i, linalg.norm(param.grad.data).item()))
        optimizer.zero_grad()
        # print("{}C: {:.2f}s".format(batch_idx, time.time()-chkpt))

        # batch_size = len(inputs)
        # step = epoch * size + (batch + 1) * batch_size
        # if batch_idx % 25 == 24:
        #     print("epoch {} batch {} at time {:.2f}".format(epoch, batch_idx+1, time.time() - start_time))
            # test(params, test_loader, model, loss_fn, epoch, batch_count=batch_idx+1)
            # model.train()

    elapsed = time.time() - start_time
    print("EPOCH {}: Time elapsed = {:.2f}, Time per batch = {:.2f}".format(epoch, elapsed, elapsed/batch_idx))

def test(params, dataloader, model, loss_fn, epoch, batch_count=0, calc_acc5=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    start_time = time.time()
    test_loss, correct, correct_top5 = 0, 0, 0

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(dataloader):
            with autocast(device_type=params.device.type):
                inputs, labels = data_batch
                inputs, labels = inputs.to(params.device), labels.to(params.device)
                pred = model(inputs)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

            if calc_acc5:
                _, pred_top5 = pred.topk(5, 1, largest=True, sorted=True)
                correct_top5 += pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).sum().item()

    test_loss /= num_batches
    accuracy = 100 * correct / size
    top5_accuracy = 100 * correct_top5 / size if calc_acc5 else None

    print(f"Test Results - Epoch {epoch+1}: Accuracy={accuracy:.2f}%, Avg loss={test_loss:.6f}")
    if calc_acc5:
        print(f"Top-5 Accuracy={top5_accuracy:.2f}%")
    print("Test time: {:.2f}".format(time.time() - start_time))

if __name__ == "__main__":
    params = Params()

    train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                                transforms.RandomHorizontalFlip(0.5),
                                # v2.ElasticTransform(alpha=30.0, sigma=3.0),
                                # v2.RandomPerspective(),
                                # v2.RandomAffine(30, translate=(0.1, 0.1)),
                                transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
        ])
    test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize(size=256, antialias=True),
                                transforms.CenterCrop(224),
                                transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    trainset = datasets.ImageFolder(root='~/spot_aws/data/ImageNet/train',
                                            transform=train_transform)
    testset = datasets.ImageFolder(root='~/spot_aws/data/ImageNet/val',
                                            transform=test_transform)

    train_loader = torch.utils.data.DataLoader(trainset,
                                                batch_size=params.batch_size,
                                                num_workers=params.workers,
                                                shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=params.batch_size//2,
                                                num_workers=params.workers,
                                                shuffle=False)

    num_classes = len(trainset.classes)
    model = Net(classes=num_classes).to(params.device)
    model = DataParallel(model.cuda())

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=params.max_lr/params.div_factor,#params.lr,
                                momentum=params.momentum,
                                weight_decay=params.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = params.epochs * steps_per_epoch
    print("Steps per epoch: {}".format(steps_per_epoch))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.lr_step_size, gamma=params.lr_gamma)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=params.max_lr,
                                                    total_steps=total_steps,
                                                    pct_start=params.pct_start,
                                                    div_factor=params.div_factor,
                                                    final_div_factor=params.final_div_factor)

    for epoch in range(50):
        train(params, train_loader, test_loader, model, loss_fn, optimizer, epoch=epoch)
        test(params, test_loader, model, loss_fn, epoch, optimizer)