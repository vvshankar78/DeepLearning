import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import cv2


def has_cuda():
    return torch.cuda.is_available()


def which_device():
    return torch.device("cuda" if has_cuda() else "cpu")


def init_seed(args):
    torch.manual_seed(args.seed)

    if has_cuda():
        print("CUDA Available")
        torch.cuda.manual_seed(args.seed)


def show_model_summary(model, input_size):
    print(summary(model, input_size=input_size))


def imshow(img, title=""):
    img = denormalize(img)
    npimg = img.numpy()
    fig = plt.figure(figsize=(15, 7))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)


def normalize(tensor, mean=[0.4919, 0.4827, 0.4472],
              std=[0.2470, 0.2434, 0.2616]):
    single_img = False
    if tensor.ndimension() == 3:
        single_img = True
        tensor = tensor[None, :, :, :]

    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    ret = tensor.sub(mean).div(std)
    return ret[0] if single_img else ret


def denormalize(tensor, mean=[0.4919, 0.4827, 0.4472],
                std=[0.2470, 0.2434, 0.2616]):
    single_img = False
    if tensor.ndimension() == 3:
        single_img = True
        tensor = tensor[None, :, :, :]

    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    ret = tensor.mul(std).add(mean)
    return ret[0] if single_img else ret


def training_curves(history):
    plt.rcParams['figure.figsize'] = [30, 12]
    plt.figure(1)
    plt.suptitle("Combined Training Curves", fontsize=28)

    plt.subplot(2, 2, 1)
    for i in history:
        plt.plot(
            np.array(history[i]['train_loss']), label=i)
    #plt.plot(np.array(hist_train_acc), 'r')
    plt.ylim(ymin=0)
    plt.ylabel('loss')
    plt.minorticks_on()
    plt.grid()
    plt.legend()
    plt.title("Training loss per batch")

    plt.subplot(2, 2, 2)
    for i in history:
        plt.plot(np.array(history[i]['test_loss']), label=i)
    plt.ylim(ymin=0)
    plt.ylabel('loss')
    plt.minorticks_on()
    plt.grid()
    plt.legend()
    plt.title("Test loss per batch")

    plt.subplot(2, 2, 3)
    for i in history:
        plt.plot(
            np.array(history[i]['train_accuracy']), label=i)
    plt.ylim(top=100)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.minorticks_on()
    plt.grid()
    plt.legend()
    plt.title("Training accuracy per batch")

    plt.subplot(2, 2, 4)
    for i in history:
        plt.plot(
            np.array(history[i]['test_accuracy']), label=i)
    plt.ylim(top=100)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.minorticks_on()
    plt.grid()
    plt.legend()
    plt.title("Test accuracy per epoch")
