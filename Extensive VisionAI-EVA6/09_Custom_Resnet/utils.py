import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib

import sys, os

my_path_cam = '/content/pytorch-grad-cam'
sys.path.append(my_path_cam)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image


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


def imshow1(img):
    img = denormalize(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.grid(b=None)
    return img


def show_misclassified(model, test_loader, device, classes, num_of_images=10):
    model.eval()
    figure = plt.figure(figsize=(20, 12))
    # num_of_images = 20
    index = 1

    misclass_img_list = []
    misclass_img_category = []
    untrans_img = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(
                device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            act = target.view_as(pred)
            # since most of the bool vec is true (good problem to have) and switch (flip) the true to false and vice versa
            bool_vec = ~pred.eq(act)

            # now extract the index number from the tensor which has 'true'
            idx = list(
                np.where(bool_vec.cpu().numpy())[0])

            if idx:  # if not a blank list
                idx_list = idx
                # print(data[idx_list[0]].shape)
                if index < num_of_images+1:
                    plt.subplot(4, 5, index)
                    plt.axis('off')
                    titl = 'act/pred : ' + \
                        str(classes[target[idx[0]].cpu().item(
                        )]) + '/' + str(classes[pred[idx[0]].cpu().item()])
                    # prints the 1st index of each batch.

                    img = data[idx[0]].cpu()
                    untrans_img.append(img)
                    image = imshow1(img)
                    misclass_img_list.append(image)
                    misclass_img_category.append(
                        target[idx[0]].cpu().item())

                    plt.title(titl)
                    index += 1
    # plt.show();

    return misclass_img_list, misclass_img_category


def save_misclassified_img(misclass_img_list):
    mis_img_name_list = []

    for i in range(len(misclass_img_list)):
        # img_name = 'images/'+'mis_img'+str(i+1)+'.jpeg'
        img_name = 'mis_img'+str(i+1)+'.jpeg'
        mis_img_name_list.append(img_name)
        image = (
            255*torch.swapaxes(misclass_img_list[i], 0, 2))
        image = torch.swapaxes(image, 0, 1)
        image = image.type(torch.uint8).cpu().numpy()
        # print(image.shape)
        matplotlib.image.imsave(img_name, image)
        # print(img_name)


def Generate_CAM(model, misclass_img_category, misclass_count, cuda):
    target_layer0 = model.layer1[0]  # RF 7x7
    # rest are the end of respective layers
    target_layer1 = model.layer1[-1]
    target_layer2 = model.layer2[-1]
    target_layer3 = model.layer3[-1]
    target_layer4 = model.layer4[-1]

    visualization = []  # store all the grad cam images here
    rgb_img_list = []

    missclassified_count = misclass_count

    for i in range(0, missclassified_count-1):

        # image_path = 'images/mis_img' + str(i+1) + '.jpeg'
        image_path = 'mis_img' + str(i+1) + '.jpeg'
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        # rgb_img = cv2.resize(rgb_img, (224, 224))    #uncomment this to scale to 224x224
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(
            rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb_img_list.append(rgb_img)

        target_category = misclass_img_category[i]
        visualization.append(rgb_img)

        for layer in [target_layer0, target_layer1, target_layer2, target_layer3, target_layer4]:
            cam = GradCAM(
                model=model, target_layer=layer, use_cuda=cuda)
            grayscale_cam = cam(
                input_tensor=input_tensor, target_category=target_category)
            grayscale_cam = grayscale_cam[0, :]

            visualization.append(
                show_cam_on_image(rgb_img, grayscale_cam))
    return visualization


def CAM_show_image(visualization, start_image):

    plt.rcParams['figure.figsize'] = [24, 12]
    plt.figure(1)
    cnt = 1
    layer_no = 0
    # start_image = 48
    for i in range(1, 31):
        # plt.subplot(int(len(visualization)/6), 6, i)
        plt.subplot(5, 6, i)
        plt.imshow(
            visualization[start_image], interpolation='bilinear')
        start_image += 1
        plt.axis('off')
        if cnt == 1:
            plt.title('misclassified image')
        else:
            plt.title('layer' + str(layer_no))
            layer_no += 1
        if cnt == 6:
            cnt = 0
            layer_no = 0
        cnt += 1
