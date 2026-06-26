from torch import is_tensor, float32, Generator
from torch.utils.data import Dataset, random_split, DataLoader

from torchvision.utils import make_grid
from torchvision.transforms import v2
from torchvision import tv_tensors

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

"""
CHARSET and char_to_idx written by Copilot.

CHARSET is a tuple of strings of characters

char_to_idx is a dictionary using characters as keys and its translation to the output layer of
the neural networks.
"""
CHARSET = (
    [chr(i) for i in range(65, 91)] +      # A–Z
    [chr(i) for i in range(97, 123)] +     # a–z
    [chr(i) for i in range(48, 58)] +      # 0–9
    ['@', '#', '$', '%', '&', '+', '?', '<', '>']
)
char_to_idx = {c: i for i, c in enumerate(CHARSET)}

class CharacterDataset(Dataset):
    """
    Character img dataset.
    https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Attributes:
        root_dir: A filepath to a folder with images
        images: Filename and label of each image in root_dir
        transform: The transform applied to an image
    """
    
    def __init__(self, root_dir: str, transform=None):
        """
        Initialises dataset.
        
        Keyword arguments:
            root_dir: A filepath to a folder with images
            transform: The transform applied to an image
        """

        # If root_dir does not exist it will raise an exception to the developer (not user).
        if not os.path.exists(root_dir):
            raise Exception(f"Root directory: {root_dir} does not exist")

        # Set class attributes
        self.root_dir = root_dir
        self.transform = transform

        # Loops through each filepath in the root_dir and saves the filename and label.
        self.images = [{
            "filename": filepath, 
            "label": char_to_idx[chr(int(filepath.removesuffix(".png").split("_")[2]))]
            } for filepath in os.listdir(root_dir)]

    def __len__(self):
        """Returns amount of items in self.images"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns grayscale image and label in format:
            (image, label)
        """
        
        # If idx is tensor it will turn it into a list
        if is_tensor(idx):
            idx = idx.tolist()

        # Joins together root directory and filename to get filepath
        filepath = os.path.join(self.root_dir, self.images[idx]["filename"])

        # Opens filepath as an image
        with Image.open(filepath) as image:
            image = image.convert("L") # Makes image grayscale

            # Applies transform to image if applicable
            if self.transform:
                image = self.transform(image)

        return image, self.images[idx]["label"]
    
class TrainTestLoader:
    """
    Utility class for CharacterDataset to provide efficent data loading.

    Attributes:
        dataset: CharacterDataset object
        trainset: Training dataset
        testset: Testing dataset
        trainloader: Dataloader for trainset
        testloader: Dataloader for testloader
        batch_size: Number of data samples processed in a single iteration
        train_totalbatches: Number of batches within trainloader
        test_totalbatches: Number of batches within testloader
    """
    def __init__(self, root_dir: str, seed=42, split=0.2, batch_size=256):
        """
        Initialise TrainTestLoader.

        Keyword arguments:
            root_dir: Filepath where dataset is stored
        
        Optional keyword arguments:
            seed: Starting point for RNG
            split: Fraction representing how much of the dataset is split into the trainset
            batch_size: Number of data samples processed in a single iteration
        """
        
        # Transform for images. Converst PIL image, numpy array, or tensor into Image tensor 
        # and normalises pixel values to be from 0 to 1.
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(float32, scale=True)
        ])

        # Create Generator object to manage RNG
        generator = Generator().manual_seed(seed)

        # Create dataset and split it into train and test sets
        self.dataset = CharacterDataset(root_dir=root_dir, transform=transform)
        self.trainset, self.testset = random_split(self.dataset, [1-split, split], generator=generator)

        # Create dataloaders
        if split != 1:
            self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        else:
            self.trainloader = []
        
        self.testloader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Batch size reference for other programs
        self.batch_size = batch_size

        self.train_totalbatches = len(self.trainloader)
        self.test_totalbatches = len(self.testloader)

class FullLoader:
    """
    Utility class for CharacterDataset to provide efficent data loading.

    Attributes:
        dataset: CharacterDataset object
        testset: Testing dataset
        loader: Dataloader for testset
        batch_size: Number of data samples processed in a single iteration
        totalbatches: Number of batches within loader
    """
    def __init__(self, root_dir: str, batch_size=256):
        """
        Initialise TrainTestLoader.

        Keyword arguments:
            root_dir: Filepath where dataset is stored
        
        Optional keyword arguments:
            batch_size: Number of data samples processed in a single iteration
        """
        
        # Transform for images. Converst PIL image, numpy array, or tensor into Image tensor 
        # and normalises pixel values to be from 0 to 1.
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(float32, scale=True)
        ])

        # Create dataset and then directly convert it into dataloader
        self.dataset = CharacterDataset(root_dir=root_dir, transform=transform)
        trainset, testset = random_split(self.dataset, [0, 1])
        self.loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Batch size reference for other programs
        self.batch_size = batch_size

        self.totalbatches = len(self.loader)

def imshow(img: tv_tensors.Image):
    """
    Takes a PyTorch image tensor and displays it using matplotlib.
    """
    
    img = img / 2 + 0.5 # Undo normalisation
    npimg = img.numpy() # Convert PyTorch tensor to numpy array
    
    # Displays image
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    ttl = TrainTestLoader(root_dir="character_images_no_noise")

    print(f"Amount of items in dataset: {len(ttl.dataset)}")
    dataiter = iter(ttl.trainloader)
    images, labels = next(dataiter)
    imshow(make_grid(images))
    print(" ".join(f"{labels[j].item():5}" for j in range(ttl.batch_size)))