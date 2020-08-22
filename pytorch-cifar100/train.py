#!/usr/bin/env	python3

import os
import sys
import argparse
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from datetime import datetime
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR


def curriculum_img_order():
    # List to store final loss of 50,000 images
    idx2Scores = []

    # Obtain the ordering
    with torch.no_grad():
        net.eval()

        for image, label in curriculum_dataloader:
            label = label.cuda()
            image = image.cuda()

            output = net(image)
            loss = cl_loss_function(output, label)
            idx2Scores.append(loss.item())

    # Pickle idx2Scores so that it can be read in utils.py to order the dataset
    with open('cifar100_img_losses', 'wb') as fp:
        pickle.dump(idx2Scores, fp)

    return idx2Scores


def train(epoch, pretrained, train_loss_list, train_acc_list, test_loss_list, test_acc_list):
    start = time.time()
    net.train()

    # Initialise variables to store plotting information
    train_loss_epoch = 0.0
    train_acc_epoch = 0.0
    correct = 0.0

    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # To obtain training accuracy
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        # Store training loss and accuracy
        train_loss_epoch += loss.item()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

    # Obtain the losses of each individual image in the dataset
    # Only execute when running the pretrained network with last layer being trained AND
    # When last epoch is being run
    if (epoch == settings.EPOCH - 1) and pretrained:
        curriculum_img_order()

    finish = time.time()

    # Store the values of the training loss and accuracy
    train_loss_epoch = train_loss_epoch / len(cifar100_training_loader.dataset)
    train_acc_epoch = correct.float() / len(cifar100_training_loader.dataset)

    train_loss_list.append(train_loss_epoch)
    train_acc_list.append(train_acc_epoch.item())

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch, train_loss_list, train_acc_list, test_loss_list, test_acc_list):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    avg_test_loss = 0.0

    for (images, labels) in cifar100_test_loader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()

    print('Evaluating Network.....')
    print('[TEST SET] Avg. Loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(cifar100_test_loader),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))

    avg_test_loss = test_loss / len(cifar100_test_loader.dataset)
    test_accuracy = correct.float() / len(cifar100_test_loader.dataset)

    test_loss_list.append(avg_test_loss)
    test_acc_list.append(test_accuracy.item())

    return correct.float() / len(cifar100_test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-curriculum', type=bool, default=False, help='curriculum initialisation')
    parser.add_argument('-pretrained', type=bool, default=False, help='train only last layer')
    args = parser.parse_args()

    net = get_network(args)

    if args.pretrained:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the location of the pretrained weights here
        weights_path = './checkpoint/resnet50/Saturday_22_August_2020_16h_31m_22s/resnet50-95-best.pth'
        net.load_state_dict(torch.load(weights_path), args.gpu)

        for param in net.parameters():
            param.requires_grad = False

        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 100)

        net = net.to(device)

        # Optimize only last layer (net.classifier[6] for vgg models, net.fc for resnet models)
        optimizer = optim.SGD(net.fc.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    else:
        # Optimize the entire network
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Data preprocessing (Train and Test DataLoaders)
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=False,
        curriculum=args.curriculum
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    # Curriculum DataLoader
    if args.pretrained:
        curriculum_dataloader = get_training_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            num_workers=4,
            batch_size=1,
            shuffle=False,
            curriculum=False
        )

        # Loss Function for Curriculum Ordering
        cl_loss_function = nn.CrossEntropyLoss(reduction='none')

    loss_function = nn.CrossEntropyLoss()
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # Create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    # Create lists to store the loss and accuracies for both Train and Test
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch, args.pretrained, train_loss_list, train_acc_list, test_loss_list, test_acc_list)
        acc = eval_training(epoch, train_loss_list, train_acc_list, test_loss_list, test_acc_list)

        # Start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    # Print only for graphing purpose
    print("\nTraining Loss: ", train_loss_list)
    print("\nTest Loss: ", test_loss_list)
    print("\nTraining Accuracy: ", train_acc_list)
    print("\nTest Accuracy: ", test_acc_list)
