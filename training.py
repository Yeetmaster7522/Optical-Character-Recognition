import torch
from torchvision.utils import make_grid
from torchvision.transforms import v2
from torch.utils.data import Dataset, random_split, DataLoader
import torch.optim as optim
from torch import nn

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import os

from models import ocr_v1 as model

CHARSET = (
    [chr(i) for i in range(65, 91)] +      # A–Z
    [chr(i) for i in range(97, 123)] +     # a–z
    [chr(i) for i in range(48, 58)] +      # 0–9
    ['@', '#', '$', '%', '&', '+', '?', '<', '>']
)
char_to_idx = {c: i for i, c in enumerate(CHARSET)}

class CharacterDataset(Dataset):
    """
    Character img dataset
    https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    
    def __init__(self, root_dir: str, transform=None):
        """
        root_dir: directory with all the images
        transform: applied on a sample
        """
        if not os.path.exists(root_dir):
            raise Exception(f"Root directory: {root_dir} does not exist")

        self.root_dir = root_dir
        self.transform = transform

        self.images = [{
            "filename": filepath, 
            "label": char_to_idx[chr(int(filepath.split("_")[2]))]
            } for filepath in os.listdir(root_dir)]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filepath = os.path.join(self.root_dir, self.images[idx]["filename"])
        image = Image.open(filepath)
        image = image.convert("L") # grayscale

        if self.transform:
            image = self.transform(image)

        return image, self.images[idx]["label"]
    

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
])

batch_size = 4

dataset = CharacterDataset(root_dir="character_images", transform=transform)
generator = torch.Generator().manual_seed(42)
trainset, testset = random_split(dataset, [0.8, 0.2], generator=generator)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


if __name__ == "__main__":
    # dataiter = iter(trainloader)
    # images, labels = next(dataiter)
    # imshow(make_grid(images))
    # print(" ".join(f"{labels[j].item():5}" for j in range(batch_size)))
    print(len(dataset))

    net = model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

    path = "model_1.pt"
    torch.save(net.state_dict(), path)