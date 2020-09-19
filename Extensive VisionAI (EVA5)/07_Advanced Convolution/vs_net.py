import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,16,3,1,1),nn.BatchNorm2d(16),nn.ReLU(),nn.Dropout(p=0.1),
                                   nn.Conv2d(16,32,3,1,1,dilation=1),nn.BatchNorm2d(32),nn.ReLU(),nn.Dropout(p=0.1)
                                   )
        
        self.pool = nn.MaxPool2d(2, 2)
       
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Dropout(p=0.1),
                                   nn.Conv2d(64,128,3,1,1,dilation=2),nn.BatchNorm2d(128),nn.ReLU(),nn.Dropout(p=0.1)
                                   ) # dilation =2
        
        self.conv111 = nn.Sequential(nn.Conv2d(128,64,1),nn.BatchNorm2d(64),nn.ReLU())
        
        self.conv3 = nn.Sequential(nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(),nn.Dropout(p=0.1),
                                   nn.Conv2d(128,256,3,1,1),nn.BatchNorm2d(256),nn.ReLU(),nn.Dropout(p=0.1)
                                   )
        
        self.conv112 = nn.Sequential(nn.Conv2d(256,64,1),nn.BatchNorm2d(64),nn.ReLU())
        
        self.conv4 = nn.Sequential(nn.Conv2d(64,128,3,1,1,dilation=1),nn.BatchNorm2d(128),nn.ReLU(),nn.Dropout(p=0.1))
        #self.conv5 = nn.Sequential(nn.Conv2d(128,256,3,1,1,dilation=1),nn.BatchNorm2d(256),nn.ReLU(),nn.Dropout(p=0.1))

        self.depthwise = nn.Sequential(nn.Conv2d(128,128,3,padding=1,groups=128),
                                       nn.Conv2d(128,256,1),
                                       nn.BatchNorm2d(256),nn.ReLU(),nn.Dropout(p=0.1)) # Depthwise
        
                                  

        self.conv113 = nn.Sequential(nn.Conv2d(256,10,1),nn.BatchNorm2d(10),nn.ReLU())

        self.gap = nn.AvgPool2d(kernel_size=3)
        
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.conv111(self.pool(self.conv2(x)))      
        
        x = self.conv112(self.pool(self.conv3(x)))
        x = self.conv113(self.depthwise((self.conv4(x))))
        x = self.gap(x)
        x = x.view(-1, 10 * 1 * 1)
        x = F.log_softmax(x,dim=-1)
        return x