## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
       
        self.pool = nn.MaxPool2d(2, 2)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)    

        self.fc1 = nn.Linear(43264, 2000)
        self.fc2 = nn.Linear(2000, 136)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        


        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn3(x)
        x = self.dropout3(x)

        x = self.pool(F.relu(self.conv4(x)))
        x = self.bn4(x)
        x = self.dropout4(x)
      
        x = x.view(x.size(0), -1)
        
        x = self.dropout5(x)
        x = F.relu(self.fc1(x))
        x = self.dropout6(x) 
        x = self.fc2(x)
        
      
        return x
