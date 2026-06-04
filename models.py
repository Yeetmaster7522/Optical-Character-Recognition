from torch import nn, flatten
import torch.nn.functional as F

class ocr_v1(nn.Module):
    """
    https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    https://www.ibm.com/think/topics/convolutional-neural-networks

    28x28 images
    all uppercase and lowercase english letters, digit 0-9, @ # $ % & + ? < >
    71 output neurons
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 71)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x