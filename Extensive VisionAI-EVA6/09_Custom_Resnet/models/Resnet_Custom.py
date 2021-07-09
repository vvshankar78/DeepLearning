import torch.nn as nn
import torch.nn.functional as F


class Net2(nn.Module):


    def ResBlock(self, in_features, out_features, pading=1):
      # convolution
      layers = []
      layers = [nn.Conv2d(in_features, out_features, 3, padding=pading, bias=False),nn.BatchNorm2d(out_features), nn.ReLU(),
                nn.Conv2d(out_features, out_features, 3, padding=pading, bias=False),nn.BatchNorm2d(out_features), nn.ReLU()]
      return nn.Sequential(*layers)

    def max_pool_block(self, in_features, out_features, pading=1):
        layers = []
        layers = [nn.Conv2d(in_features, out_features, 3, padding=pading, bias=False), nn.MaxPool2d(2,2), nn.BatchNorm2d(out_features), nn.ReLU()]
        return nn.Sequential(*layers)


    def __init__(self):
        super(Net2, self).__init__()
        self.convblock0 = nn.Sequential(nn.Conv2d(3,64, 3, padding=1, bias=False),nn.BatchNorm2d(64), nn.ReLU()) #38
        self.pool1 = self.max_pool_block(64,128)  #19
        self.convblock1 = self.ResBlock(128,128) #19
        self.pool2 = self.max_pool_block(128,256)  #9   
        self.pool3 = self.max_pool_block(256,512)  #4             
        self.convblock2 = self.ResBlock(512,512) #4
        

        # self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=4)) # output_size = 1
        self.max_pool = nn.MaxPool2d(4,4)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.convblock0(x)
        x = self.pool1(x)
        x1 = self.convblock1(x)
        x = x+ x1
        x = self.pool2(x)
        x = self.pool3(x)
        x2 = self.convblock2(x)
        x = x + x2

        
        # print(x.shape)
        # x = self.gap(x)
        x = self.max_pool(x)
        # print(x.shape)
        x = x.view(-1,512)
        x = self.fc1(x)
        x = x.view(-1,10)
        return F.log_softmax(x, dim=-1)