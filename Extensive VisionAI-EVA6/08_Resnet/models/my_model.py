import torch.nn as nn

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(args.dropout_value),  # In: 32x32x3 | Out: 32x32x32 | RF: 3x3

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32), # In: 32x32x32 | Out: 32x32x32 | RF: 5x5
        )
        self.pool1 = nn.MaxPool2d(2, 2) # In: 32x32x32 | Out: 16x16x32 | RF: 6x6
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(args.dropout_value),  # In: 16x16x32 | Out: 16x16x64 | RF: 10x10

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64), # In: 16x16x64 | Out: 16x16x64 | RF: 14x14
        )
        self.pool2 = nn.MaxPool2d(2, 2) # In: 16x16x64 | Out: 8x8x64 | RF:16x16
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, groups=64, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(args.dropout_value),  # In: 8x8x64 | Out: 8x8x64 | RF: 24x24

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64), # In: 8x8x64 | Out: 8x8x64 | RF: 32x32
        )
        self.pool3 = nn.MaxPool2d(2, 2) # In: 8x8x64 | Out: 4x4x64 | RF: 36x36
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(args.dropout_value),  # In: 4x4x64 | Out: 4x4x128 | RF: 68x68

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),  # In: 4x4x128 | Out: 4x4x128 | RF: 84x84
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)  # In: 4x4x128 | Out: 1x1x128 | RF: 108x108
        self.layer5 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10),
            # nn.ReLU() NEVER!
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.view(-1, 128)
        x = self.layer5(x)
        return x
    


class Net1(nn.Module):
    def __init__(self, args):
        super(Net1, self).__init__()
        dropout_value = args.dropout_value
        
        layer1_channel = 32
        layer2_channel = 32
        layer3_channel = 64
        layer4_channel = 128
        
        
        # # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=layer1_channel, kernel_size=(3, 3), padding='same', bias=False, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(layer1_channel),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=layer1_channel, out_channels=layer1_channel, kernel_size=(3, 3), padding='same', bias=False, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(layer1_channel),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=layer1_channel, out_channels=layer1_channel, kernel_size=(3, 3), stride=2, padding=1, bias=False, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(layer1_channel),
            nn.Dropout(dropout_value)
        )

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            # Depthwise
            #nn.Conv2d(in_channels=layer1_channel, out_channels=layer1_channel, kernel_size=(3, 3), padding=1, groups=layer1_channel, bias=False, dilation=2),
            #nn.Conv2d(layer1_channel,layer2_channel,1),
            nn.Conv2d(in_channels=layer1_channel, out_channels=layer2_channel, kernel_size=(3, 3), padding='same', bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(layer2_channel),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=layer2_channel, out_channels=layer2_channel, kernel_size=(3, 3), stride=2, padding=1, bias=False, dilation = 1),
            nn.ReLU(),
            nn.BatchNorm2d(layer2_channel),
            nn.Dropout(dropout_value)
        )

        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            #nn.Conv2d(in_channels=layer2_channel, out_channels=layer3_channel, kernel_size=(3, 3), padding='same', bias=False, dilation=2),
            # Depthwise
            nn.Conv2d(in_channels=layer2_channel, out_channels=layer2_channel, kernel_size=(3, 3), padding=1, groups=layer2_channel, bias=False, dilation=2),
            nn.Conv2d(layer2_channel,layer3_channel,1),
            nn.ReLU(),
            nn.BatchNorm2d(layer3_channel),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=layer3_channel, out_channels=layer3_channel, kernel_size=(3, 3), stride=2, padding=2, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(layer3_channel),
            nn.Dropout(dropout_value)
        )
        
        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels=layer3_channel, out_channels=layer3_channel, kernel_size=(3, 3), padding=2, groups=layer3_channel, bias=False, dilation=2),
            nn.Conv2d(layer3_channel,layer4_channel,1),
            nn.ReLU(),
            # Depthwise
            nn.Conv2d(in_channels=layer4_channel, out_channels=layer4_channel, kernel_size=(3, 3), padding=2, groups=layer4_channel, bias=False, dilation=2),
            nn.Conv2d(layer4_channel,layer4_channel,1),
            nn.ReLU(),

            #to remove last fc layer
            nn.Conv2d(layer4_channel,10,1),
            #nn.ReLU()
        )

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2)
        )


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = x.view(-1, 10 * 1 * 1)
        return nn.functional.log_softmax(x, dim=-1)
    
 
class Net2(nn.Module):
    def __init__(self, args):
        super(Net2, self).__init__()
        dropout_value = args.dropout_value
        
        layer1_channel = 32
        layer2_channel = 32
        layer3_channel = 64
        layer4_channel = 128
        
        
        # # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=layer1_channel, kernel_size=(3, 3), padding='same', bias=False, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(layer1_channel),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=layer1_channel, out_channels=layer1_channel, kernel_size=(3, 3), padding='same', bias=False, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(layer1_channel),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=layer1_channel, out_channels=layer1_channel, kernel_size=(3, 3), stride=1, padding=1, bias=False, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(layer1_channel),
            nn.Dropout(dropout_value)
        )

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            # Depthwise
            #nn.Conv2d(in_channels=layer1_channel, out_channels=layer1_channel, kernel_size=(3, 3), padding=1, groups=layer1_channel, bias=False, dilation=2),
            #nn.Conv2d(layer1_channel,layer2_channel,1),
            nn.Conv2d(in_channels=layer1_channel, out_channels=layer2_channel, kernel_size=(3, 3), stride=2, padding=2, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(layer2_channel),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=layer2_channel, out_channels=layer2_channel, kernel_size=(3, 3), stride=1, padding='same', bias=False, dilation = 1),
            nn.ReLU(),
            nn.BatchNorm2d(layer2_channel),
            nn.Dropout(dropout_value)
        )

        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            #nn.Conv2d(in_channels=layer2_channel, out_channels=layer3_channel, kernel_size=(3, 3), padding='same', bias=False, dilation=2),
            # Depthwise
            nn.Conv2d(in_channels=layer2_channel, out_channels=layer2_channel, kernel_size=(3, 3), stride=2, padding=2, groups=layer2_channel, bias=False, dilation=2),
            nn.Conv2d(layer2_channel,layer3_channel,1),
            nn.ReLU(),
            nn.BatchNorm2d(layer3_channel),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=layer3_channel, out_channels=layer3_channel, kernel_size=(3, 3), stride=1, padding='same', bias=False, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(layer3_channel),
            nn.Dropout(dropout_value)
        )
        
        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels=layer3_channel, out_channels=layer3_channel, kernel_size=(3, 3), stride=2, padding=2, groups=layer3_channel, bias=False, dilation=2),
            nn.Conv2d(layer3_channel,layer4_channel,1),
            nn.ReLU(),
            nn.BatchNorm2d(layer4_channel),
            nn.Dropout(dropout_value),
            # Depthwise
            nn.Conv2d(in_channels=layer4_channel, out_channels=layer4_channel, kernel_size=(3, 3), stride=1, padding='same', groups=layer4_channel, bias=False, dilation=1),
            nn.Conv2d(layer4_channel,layer4_channel,1),
            nn.ReLU(),
            nn.BatchNorm2d(layer4_channel),
            nn.Dropout(dropout_value),

            #to remove last fc layer
            nn.Conv2d(layer4_channel,10,1),
            #nn.ReLU()
        )

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2)
        )


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3] )
        return nn.functional.log_softmax(x, dim=-1)