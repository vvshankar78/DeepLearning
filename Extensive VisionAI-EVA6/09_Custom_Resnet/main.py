
from __future__ import print_function
import numpy as np
import argparse



import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
import matplotlib
from torch_lr_finder import LRFinder
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import StepLR


from config import ModelConfig
import sys, os

from models import Resnet
from models import my_model
from models import Resnet_Custom
from train import train
from test1 import test
from config import ModelConfig
from utils import *
from data.data_engine import DataEngine


# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--epoch', default=10, type=int, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
# args1 = parser.parse_args()

# print(args1)
# print(args1.epoch)
# print(args1.resume)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

SEED = 1

args = ModelConfig()
args.dropout_value = 0.0


def get_data():
  # View model config
  args.print_config()
  print(args.epochs)

  data = DataEngine(args)
  train_loader= data.train_loader
  test_loader = data.test_loader
  return data, train_loader, test_loader


def show_data(data):
  data.show_samples()


def check_cuda():
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)
  print(device)

  # For reproducibility
  torch.manual_seed(SEED)

  if cuda:
      torch.cuda.manual_seed(SEED)

def get_model(get_summary=False):
  model = Resnet_Custom.Net2().to(device)
  if get_summary:
      summary(model, input_size=(3, 38, 38))
  return model


def get_lr_finder(model, train_loader):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=1e-7, weight_decay=1e-2)
  lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
  lr_finder.range_test(train_loader, end_lr=100, num_iter=100, step_mode="exp")
  lr_finder.plot()
  return


def get_ocp_plot(train_loader, model, max_lr=0.1):   
  EPOCHS = args.epochs
  peak = args.peak
  peak_pct = peak/EPOCHS
  optimizer = optim.SGD(model.parameters(), lr=1e-7, weight_decay=1e-2)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=EPOCHS,pct_start=peak_pct, anneal_strategy='linear')
  lrs = []

  for i in range(EPOCHS*len(train_loader)):
      optimizer.step()
      lrs.append(optimizer.param_groups[0]["lr"])
  #     print("Factor = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])
      scheduler.step()

  plt.plot(lrs)
  return


def run_model(model, train_loader, test_loader, max_lr=0.1):
  history = {}
  EPOCHS = args.epochs
  l1_decay=0.0
  l2_decay=0.0
  peak = args.peak # epoch you want the max lr. 
  peak_pct = peak/EPOCHS
  lrs = []

  # model = Net().to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  # scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader),
                                                  epochs=EPOCHS,
                                                  pct_start=peak_pct, 
                                                  anneal_strategy='linear')

  # run_model(model, optimizer, scheduler, EPOCHS, l1=0.0, l2=0.0)

  train_loss_list = []
  train_accuracy_list = []
  test_loss_list = []
  test_accuracy_list = []
  misclassified_imgs = []
  for epoch in range(EPOCHS):
      print("EPOCH:", epoch+1)
      train_loss_list, train_accuracy_list = train(model, device, train_loader, criterion, optimizer, epoch, l1_decay, l2_decay, train_loss_list, train_accuracy_list, scheduler)
      print("\nlearning rate", optimizer.param_groups[0]['lr'])
      # scheduler.step()
      # test(model1, device, test_loader, test_losses, test_acc, misclassified_imgs, epoch==EPOCHS-1)
      test_loss_list, test_accuracy_list, misclassified_imgs = test(model, device, test_loader, criterion, classes, test_loss_list, test_accuracy_list, misclassified_imgs, epoch==EPOCHS-1)

  name = 'Resnet18'
  history[name] = {}
  history[name]['train_loss'] = train_loss_list
  history[name]['train_accuracy'] = train_accuracy_list
  history[name]['test_loss'] = test_loss_list
  history[name]['test_accuracy'] = test_accuracy_list
  history[name]['misclassified_imgs'] = misclassified_imgs
  return history

def get_training_curves(history):
  training_curves(history)
  matplotlib.pyplot.show()

def get_show_misclassified(model, test_loader, num_of_images=20):
  misclass_img_list, misclass_img_category = show_misclassified(model, test_loader, device, classes, num_of_images)
  matplotlib.pyplot.show()
  save_misclassified_img(misclass_img_list)


# visualization = Generate_CAM(model, misclass_img_category, 20, cuda)
# CAM_show_image(visualization, 0)

# matplotlib.pyplot.show()

# CAM_show_image(visualization, 30)
# matplotlib.pyplot.show()




