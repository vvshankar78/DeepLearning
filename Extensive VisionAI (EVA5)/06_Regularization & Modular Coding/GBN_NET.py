# Model Architecture
import torch.nn as nn
import torch.nn.functional as F
import torch

# num_splits=2


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(
            num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(
            num_features * self.num_splits))

    def train(self, mode=True):
        # lazily collate stats when we are going to use them
        if (self.training is True) and (mode is False):
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H,
                           W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(
                    self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)


class Net_GBN(nn.Module):
    def __init__(self, dropout_value=0.05, num_splits=2):
        super(Net_GBN, self).__init__()
        self.dropout_value = dropout_value
        self.num_splits = num_splits
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(
                3, 3), padding=0, bias=False),
            nn.ReLU(),
            GhostBatchNorm(
                num_features=10, num_splits=self.num_splits),
            nn.Dropout(self.dropout_value)
        )  # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(
                3, 3), padding=0, bias=False),
            nn.ReLU(),
            GhostBatchNorm(
                num_features=10, num_splits=self.num_splits),
            nn.Dropout(self.dropout_value)
        )  # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(
                3, 3), padding=0, bias=False),
            nn.ReLU(),
            GhostBatchNorm(
                num_features=10, num_splits=self.num_splits),
            nn.Dropout(self.dropout_value)
        )  # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(
                1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU()
        )  # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(
                3, 3), padding=0, bias=False),
            nn.ReLU(),
            GhostBatchNorm(
                num_features=10, num_splits=self.num_splits),
            nn.Dropout(self.dropout_value)
        )  # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(
                3, 3), padding=0, bias=False),
            nn.ReLU(),
            GhostBatchNorm(
                num_features=10, num_splits=self.num_splits),
            nn.Dropout(self.dropout_value)
        )  # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(
                3, 3), padding=0, bias=False),
            nn.ReLU(),
            GhostBatchNorm(
                num_features=10, num_splits=self.num_splits),
            nn.Dropout(self.dropout_value)
        )  # output_size = 5

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(
                3, 3), padding=0, bias=False),
            nn.ReLU(),
            GhostBatchNorm(
                num_features=10, num_splits=self.num_splits),
            nn.Dropout(self.dropout_value)
        )  # output_size = 3

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        )

        # TRANSITION BLOCK 2
        # self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(
                1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU()
        )  # output_size = 5

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)
        x = self.convblock10(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
