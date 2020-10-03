import torch
import torchvision

from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import random
import numpy as np


class AlbumentationTransforms:
    """
    Helper class to create test and train transforms using Albumentations
    """

    def __init__(self, transforms_list=[]):
        transforms_list.append(AP.ToTensor())

        self.transforms = A.Compose(transforms_list)

    def __call__(self, img):
        img = np.array(img)
        # print(img)
        return self.transforms(image=img)['image']


def load(train_transform, test_transform):

    # Transformation for Training
    # train_transform = transforms.Compose(
    # [
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # # Transformation for Test
    # test_transform = transforms.Compose(
    # [transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Get the Train and Test Set
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)

    SEED = 1

    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4,
                           pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    trainloader = torch.utils.data.DataLoader(
        trainset, **dataloader_args)
    testloader = torch.utils.data.DataLoader(
        testset, **dataloader_args)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return classes, trainloader, testloader
