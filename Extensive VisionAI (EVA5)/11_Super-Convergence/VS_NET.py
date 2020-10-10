# Model Architecture
import torch.nn as nn
import torch.nn.functional as F
import torch

# dropout_value = 0.05

# This is to test out the syncing the google drive.


class Net(nn.Module):
    def __init__(self, dropout_value=0.05, input_ch=3):
        super(Net, self).__init__()
        self.dropout_value = dropout_value
        self.input_ch = input_ch

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_ch, out_channels=10, kernel_size=(
                3, 3), padding=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )  # input =3, output=10 channels

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=16, kernel_size=(
                3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 11

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(
                3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(
                3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(
                3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=224, out_channels=256, kernel_size=(
                3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=480, out_channels=512, kernel_size=(
                3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=992, out_channels=1024, kernel_size=(
                3, 3), padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.fc = nn.Sequential(nn.Linear(1024, 10))

    def forward(self, x):
        x2 = self.convblock1(x)
        x3 = self.convblock2(
            torch.cat((x, x2), dim=1))  # ip = 13, op=16
        # ch=32 size =16
        # ip 32, op32
        x4 = self.pool1(torch.cat((x, x2, x3), dim=1))
        x5 = self.convblock3(x4)  # inp 32, op=32
        x6 = self.convblock4(
            torch.cat((x4, x5), dim=1))  # ip 64, op 64
        x7 = self.convblock5(
            torch.cat((x4, x5, x6), dim=1))  # ip 128, op 128

        x8 = self.pool1(
            torch.cat((x5, x6, x7), dim=1))  # 224
        x9 = self.convblock6(x8)  # inp 224, op 256
        x10 = self.convblock7(
            torch.cat((x8, x9), dim=1))  # 480 op 512
        x11 = self.convblock8(
            torch.cat((x8, x9, x10), dim=1))  # 1024
        x12 = self.gap(x11)  # 1024
        # x = x.view(-1, 10)
        x13 = self.fc(x12)  # 10
        return F.log_softmax(x, dim=1)
