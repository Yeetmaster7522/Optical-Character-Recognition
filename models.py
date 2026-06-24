"""
https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
https://www.ibm.com/think/topics/convolutional-neural-networks
https://www.sciencedirect.com/science/article/pii/S2666720725000189

28x28 images
all uppercase and lowercase english letters, digit 0-9, @ # $ % & + ? < >
71 output neurons
"""

from torch import nn, flatten
import torch.nn.functional as F

class ocr_v1(nn.Module):
    """
    A child class of nn.Module.

    Model architecture:
        - conv(1,6,5)
        - relu
        - pool(2,2)
        - conv(6,16,5)
        - relu
        - pool(2,2)
        - flatten
        - fcl(256, 120)
        - fc2(120, 84)
        - fc3(84, 71)
    """
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) #out = (in - kernel_size) + 1
        self.pool = nn.MaxPool2d(2, 2) #out = in / 2
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 4 * 4, 120) #in_features = channels * dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 71)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #12x12
        x = self.pool(F.relu(self.conv2(x))) #4x4
        x = flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ocr_v2(nn.Module):
    """
    A child class of nn.Module.

    Model architecture:
        - conv(1,16,3)
        - relu
        - pool(2,2)
        - conv(16,32,3)
        - relu
        - pool(2,2)
        - flatten
        - fcl(800, 240)
        - fc2(240, 168)
        - fc3(168, 71)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3) #out = (in - kernel_size) + 1
        self.pool = nn.MaxPool2d(2, 2) #out = in / 2
        self.conv2 = nn.Conv2d(16, 32, 3)

        self.fc1 = nn.Linear(32 * 5 * 5, 240) #in_features = channels * dimension
        self.fc2 = nn.Linear(240, 168)
        self.fc3 = nn.Linear(168, 71)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #28-3+1=26 26/2=13
        x = self.pool(F.relu(self.conv2(x))) #13-3+1=11 11/2=5.5
        x = flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ocr_v3(nn.Module):
    """
    A child class of nn.Module.

    Model architecture:
        - conv(1,32,3)
        - relu
        - pool(2,2)
        - conv(32,64,3)
        - relu
        - pool(2,2)
        - flatten
        - fcl(1600, 240)
        - fc2(240, 168)
        - fc3(168, 71)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) #out = (in - kernel_size) + 1
        self.pool = nn.MaxPool2d(2, 2) #out = in / 2
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.fc1 = nn.Linear(64 * 5 * 5, 240) #in_features = channels * dimension
        self.fc2 = nn.Linear(240, 168)
        self.fc3 = nn.Linear(168, 71)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #28-3+1=26 26/2=13
        x = self.pool(F.relu(self.conv2(x))) #13-3+1=11 11/2=5.5
        x = flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ocr_v4(nn.Module):
    """
    A child class of nn.Module. Inspired by https://arxiv.org/pdf/1512.03385v1 
    VGG-19 model architecture.

    Model architecture:
        - conv(1,16,3, padding=1)
        - relu
        - pool(2,2)
        - conv(16,32,3, padding=1)
        - relu
        - pool(2,2)
        - flatten
        - fcl(1568, 256)
        - relu
        - dropout(0.1)
        - fc2(256, 128)
        - relu
        - dropout(0.1)
        - fc3(128, 71)
    """
    
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        #at this point I'm just gonna ask ChatGPT to calculate the input features for me...
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc3 = nn.Linear(128, 71)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class ocr_v5(nn.Module):
    def __init__(self):
        super().__init__()
