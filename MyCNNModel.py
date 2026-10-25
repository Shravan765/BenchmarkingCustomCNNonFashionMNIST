import torch
import torch.nn as nn

'''
Here, 
all the convolution kernels are 3X3 with stride = 1 and padding = 0
all the MaxPooling kernels are 3x3 with stride = 3
'''
class MyCNNModel_v1_0(nn.Module):
    def __init__(self):
        super(MyCNNModel_v1_0, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=3, ceil_mode=False)
        self.fc = nn.Linear(in_features=128, out_features=10)
    def forward(self, x): # x will be 1 x 28 x 28
        x = self.conv2d_1(x) # 32 x 26 x 26
        x = self.maxPool(x) # 32 x 8 x 8 (ceil_mode is false)
        x = torch.relu(x) # non-linearity
        x = self.conv2d_2(x) # 32 x 6 x 6
        x = self.maxPool(x) # 32 x 2 x 2 
        x = torch.relu(x) # non-linearity
        x = x.view(-1, 128) # resizing
        x = self.fc(x) # returns one among the 10 labels
        return x

# uses 64 channels for 2nd conv, rest same as v1
class MyCNNModel_v1_1(nn.Module):
    def __init__(self):
        super(MyCNNModel_v1_1, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=3, ceil_mode=False)
        self.fc = nn.Linear(in_features=256, out_features=10)
    def forward(self, x): # x will be 1 x 28 x 28
        x = self.conv2d_1(x) # 32 x 26 x 26
        x = self.maxPool(x) # 32 x 8 x 8 (ceil_mode is false)
        x = torch.relu(x) # non-linearity
        x = self.conv2d_2(x) # 64 x 6 x 6
        x = self.maxPool(x) # 64 x 2 x 2 
        x = torch.relu(x) # non-linearity
        x = x.view(-1, 256) # resizing
        x = self.fc(x) # returns one among the 10 labels
        return x

'''
Here, 
all the convolution kernels are 5X5 with stride = 1 and padding = 0
all the MaxPooling kernels are 3x3 with stride = 3
'''
class MyCNNModel_v2_0(nn.Module):
    def __init__(self):
        super(MyCNNModel_v2_0, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.maxPool = nn.MaxPool2d(kernel_size=3, ceil_mode=False)
        self.fc = nn.Linear(in_features=32, out_features=10)
    def forward(self, x): # x will be 1 x 28 x 28
        x = self.conv2d_1(x) # 32 x 24 x 24
        x = self.maxPool(x) # 32 x 8 x 8 (ceil_mode is false)
        x = torch.relu(x) # non-linearity
        x = self.conv2d_2(x) # 32 x 4 x 4
        x = self.maxPool(x) # 32 x 1 x 1 
        x = torch.relu(x) # non-linearity
        x = x.view(-1, 32) # resizing
        x = self.fc(x) # returns one among the 10 labels
        return x

# uses 64 channels for 2nd convolution
class MyCNNModel_v2_1(nn.Module):
    def __init__(self):
        super(MyCNNModel_v2_1, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=3, ceil_mode=False)
        self.fc = nn.Linear(in_features=64, out_features=10)
    def forward(self, x): # x will be 1 x 28 x 28
        x = self.conv2d_1(x) # 32 x 24 x 24
        x = self.maxPool(x) # 32 x 8 x 8 (ceil_mode is false)
        x = torch.relu(x) # non-linearity
        x = self.conv2d_2(x) # 64 x 4 x 4
        x = self.maxPool(x) # 64 x 1 x 1 
        x = torch.relu(x) # non-linearity
        x = x.view(-1, 64) # resizing
        x = self.fc(x) # returns one among the 10 labels
        return x

class MyCNNModel_v3(nn.Module):
    def __init__(self):
        super(MyCNNModel_v3, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=3 ,ceil_mode=False)
        self.fc1 = nn.Linear(in_features=256, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)

    def forward(self, x): # x will be 1 x 28 x 28
        x = self.conv2d_1(x) # 32 x 24 x 24
        x = self.maxPool(x) # 32 x 8 x 8 (ceil_mode is false)
        x = torch.relu(x) # non-linearity
        x = self.conv2d_2(x) # 64 x 6 x 6
        x = self.maxPool(x) # 64 x 2 x 2 
        x = torch.relu(x) # non-linearity
        x = x.view(-1, 64*2*2) # resizing
        x = self.fc1(x) # 256 -> 32
        x = torch.relu(x) 
        x = self.fc2(x) # returns one among the 10 labels
        return x