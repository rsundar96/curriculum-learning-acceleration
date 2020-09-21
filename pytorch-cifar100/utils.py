import sys
import pickle

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_network(args):
    """ return given network
    """

    if args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=False, curriculum=False, anti=False, weights=False, temp=1):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    if curriculum or anti:
        # Read the img losses from the pickled file
        with open ('cifar100_img_losses', 'rb') as fp:
            img_losses = pickle.load(fp)

        curriculum_ordering = np.argsort(img_losses).tolist()
        num_samples = len(cifar100_training.data)

        if anti:
            curriculum_ordering.reverse()
            shuffled_dataset = Subset(cifar100_training, curriculum_ordering)
            cifar100_training_loader = DataLoader(shuffled_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

        else:
            weights = softmax(np.asarray(img_losses) / temp)
            sampler = WeightedRandomSampler(weights, num_samples, replacement=False)

            shuffled_dataset = Subset(cifar100_training, curriculum_ordering)
            cifar100_training_loader = DataLoader(shuffled_dataset, shuffle=shuffle, sampler=sampler, num_workers=num_workers, batch_size=batch_size)

    else:
        cifar100_training_loader = DataLoader(cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
