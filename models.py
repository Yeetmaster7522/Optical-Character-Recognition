"""
https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
https://www.ibm.com/think/topics/convolutional-neural-networks
https://www.sciencedirect.com/science/article/pii/S2666720725000189

28x28 images
all uppercase and lowercase english letters, digit 0-9, @ # $ % & + ? < >
71 output neurons
"""

from torch import nn, flatten
from torch.utils.checkpoint import checkpoint_sequential



CHUNKS = 2 



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
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 6, 5), #out = (in - kernel_size) + 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2) #out = in / 2
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fcl = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120), #in_features = channels * dimension
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 71)
        )

    def forward(self, x):
        x = checkpoint_sequential(self.block1, CHUNKS, x, use_reentrant=False) #12x12
        x = checkpoint_sequential(self.block2, CHUNKS, x, use_reentrant=False) #4x4
        
        x = flatten(x, 1)
        x = checkpoint_sequential(self.fcl, CHUNKS, x, use_reentrant=False)
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
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3), #out = (in - kernel_size) + 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2) #out = in / 2
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fcl = nn.Sequential(
            nn.Linear(32 * 5 * 5, 240), #in_features = channels * dimension
            nn.ReLU(),
            nn.Linear(240, 168),
            nn.ReLU(),
            nn.Linear(168, 71)
        )

    def forward(self, x):
        x = checkpoint_sequential(self.block1, CHUNKS, x, use_reentrant=False)
        x = checkpoint_sequential(self.block2, CHUNKS, x, use_reentrant=False)

        x = flatten(x, 1)
        x = checkpoint_sequential(self.fcl, CHUNKS, x, use_reentrant=False)
        return x



class ocr_v3(nn.Module):
    """
    A child class of nn.Module.

    Model architecture:
        - conv(1,32,3)
        - pool(2,2)
        - conv(32,64,3)
        - pool(2,2)
        - linear(1600, 240)
        - linear(240, 168)
        - linear(168, 71)
    """

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3), #out = (in - kernel_size) + 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2) #out = in / 2
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fcl = nn.Sequential(
            nn.Linear(64 * 5 * 5, 240), #in_features = channels * dimension
            nn.ReLU(),

            nn.Linear(240, 168),
            nn.ReLU(),

            nn.Linear(168, 71)
        )

    def forward(self, x):
        x = checkpoint_sequential(self.block1, CHUNKS, x, use_reentrant=False) #28-3+1=26 26/2=13
        x = checkpoint_sequential(self.block2, CHUNKS, x, use_reentrant=False) #13-3+1=11 11/2=5.5

        x = flatten(x, 1)
        x = checkpoint_sequential(self.fcl, CHUNKS, x, use_reentrant=False)
        return x



class ocr_v4(nn.Module):
    """
    A child class of nn.Module. Inspired by https://arxiv.org/pdf/1512.03385v1 
    VGG-19 model architecture.

    Model architecture:
        - conv(1,16,3, padding=1)
        - pool(2,2)
        - conv(16,32,3, padding=1)
        - pool(2,2)
        - fcl(1568, 256)
        - fc2(256, 128)
        - fc3(128, 71)
    """
    
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        #at this point I'm just gonna ask ChatGPT to calculate the input features for me...
        self.fcl = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 71)
        )

    def forward(self, x):
        x = checkpoint_sequential(self.block1, CHUNKS, x, use_reentrant=False)
        x = checkpoint_sequential(self.block2, CHUNKS, x, use_reentrant=False)

        x = flatten(x, 1)
        x = checkpoint_sequential(self.fcl, CHUNKS, x, use_reentrant=False)

        return x



class ocr_v5(nn.Module):
    """
    A child class of nn.Module. Upgrade of ocr_v4

    Model architecture:
        - conv(1,16,3, padding=1)
        - pool(2,2)
        - conv(16,32,3, padding=1)
        - pool(2,2)
        - fcl(1568, 256)
        - fc2(256, 128)
        - fc3(128, 71)
    """
    
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        #at this point I'm just gonna ask ChatGPT to calculate the input features for me...
        self.fcl = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 71)
        )

    def forward(self, x):
        x = checkpoint_sequential(self.block1, CHUNKS, x, use_reentrant=False)
        x = checkpoint_sequential(self.block2, CHUNKS, x, use_reentrant=False)

        x = flatten(x, 1)
        x = checkpoint_sequential(self.fcl, CHUNKS, x, use_reentrant=False)

        return x



class ocr_wingding(nn.Module):
    """
    A child class of nn.Module. Similar architecture to ocr_v5

    Model architecture:
        - conv(1,16,3, padding=1)
        - pool(2,2)
        - conv(16,32,3, padding=1)
        - pool(2,2)
        - fcl(1568, 256)
        - fc2(256, 128)
        - fc3(128, 52)
    """
    
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        #at this point I'm just gonna ask ChatGPT to calculate the input features for me...
        self.fcl = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 52)
        )

    def forward(self, x):
        x = checkpoint_sequential(self.block1, CHUNKS, x, use_reentrant=False)
        x = checkpoint_sequential(self.block2, CHUNKS, x, use_reentrant=False)

        x = flatten(x, 1)
        x = checkpoint_sequential(self.fcl, CHUNKS, x, use_reentrant=False)

        return x