from torch import is_tensor, float32, Generator
from torchvision.utils import make_grid
from torchvision.transforms import v2
from torch.utils.data import Dataset, random_split, DataLoader

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import os

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
        if is_tensor(idx):
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
    v2.ToDtype(float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
])

batch_size = 4

dataset = CharacterDataset(root_dir="character_images", transform=transform)
generator = Generator().manual_seed(42)
trainset, testset = random_split(dataset, [0.8, 0.2], generator=generator)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


if __name__ == "__main__":
    print(f"Amount of items in dataset: {len(dataset)}")
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(make_grid(images))
    print(" ".join(f"{labels[j].item():5}" for j in range(batch_size)))