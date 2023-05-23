import argparse

import numpy as np
import time
import random

import torch
import torchvision
import torchvision.transforms as transforms

from mlp import MLP

random.seed(int(time.time()))
np.random.seed(int(time.time()))

def main(args):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    train_dl = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers, shuffle=True)

    mlp = MLP(train_dl, test_dl, args.epochs, args.lr, args.gamma, args.initialization,args.hidden_nodes, args.log_name)

    mlp.train(args.optimizer, args.activation, gradient_check=False)
    mlp.plot_test()
    mlp.plot_loss()

def parse_args():
    parser = argparse.ArgumentParser(description='Numply MLP example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='input batch size for training (default: 512)')
    parser.add_argument('--num-workers', type=int, default=10, metavar='N', help='number of worker of torch to train (default: 10)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--initialization', type=str, default='He', metavar='STR', help='Initialization method (default: Xavier)')
    parser.add_argument('--optimizer', type=str, default='SGD', metavar='STR', help='Optimizer (default: Adam)')
    parser.add_argument('--activation', type=str, default='ReLU', metavar='STR', help='Activation function (default: Tanh)')
    parser.add_argument('--hidden-nodes', type=int, default=64, metavar='N', help='Hidden nodes (default: 20)')
    parser.add_argument('--log-name', type=str, default='log', metavar='STR', help='Log name (default: log)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)