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
        self.__root_dir = root_dir
        self.__transform = transform

        # Loops through each filepath in the root_dir and saves the filename and label.
        self.__images = [{
            "filename": filepath, 
            "label": char_to_idx[chr(int(filepath.removesuffix(".png").split("_")[2]))]
            } for filepath in os.listdir(root_dir)]


    def __len__(self):
        """Returns amount of items in self.images"""
        return len(self.__images)


    def __getitem__(self, idx):
        """
        Returns grayscale image and label in format:
            (image, label)
        """
        
        # If idx is tensor it will turn it into a list
        if is_tensor(idx):
            idx = idx.tolist()

        # Joins together root directory and filename to get filepath
        filepath = os.path.join(self.__root_dir, self.__images[idx]["filename"])

        # Opens filepath as an image
        with Image.open(filepath) as image:
            image = image.convert("L") # Makes image grayscale

            # Applies transform to image if applicable
            if self.__transform:
                image = self.__transform(image)

        return image, self.__images[idx]["label"]
    
    @property
    def images(self):
        return self.__images

class BaseLoader:
    """
    Base for utility classes for CharacterDataset to provide efficient data loading.
    """

    def __init__(self, root_dir: str, seed):
        """
        Initialise Loader.

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
        self.__generator = Generator().manual_seed(seed)


        # Create dataset and split it into train and test sets
        self.__dataset = CharacterDataset(root_dir=root_dir, transform=transform)


    @property
    def dataset(self):
        return self.__dataset
    
    @property
    def batch_size(self):
        return self.__batch_size

class TrainTestLoader(BaseLoader):
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

        super().__init__(root_dir, seed)
        self.__trainset, self.__testset = random_split(
            self.dataset, 
            [1-split, split], 
            generator=self.generator
            )


        # Create dataloaders
        self.__trainloader = DataLoader(
            self.__trainset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
            )        
        self.__testloader = DataLoader(
            self.__testset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
            )


        # Batch size reference for other programs
        self.__batch_size = batch_size

        self.__train_totalbatches = len(self.__trainloader)
        self.__test_totalbatches = len(self.__testloader)


    @property
    def trainset(self):
        return self.__trainset
    
    @property
    def testset(self):
        return self.__testset
    
    @property
    def trainloader(self):
        return self.__trainloader
    
    @property
    def testloader(self):
        return self.__testloader

    @property
    def train_totalbatches(self):
        return self.__train_totalbatches
    
    @property
    def test_totalbatches(self):
        return self.__test_totalbatches



class FullLoader(BaseLoader):
    """
    Utility class for CharacterDataset to provide efficent data loading.

    Attributes:
        dataset: CharacterDataset object
        testset: Testing dataset
        loader: Dataloader for testset
        batch_size: Number of data samples processed in a single iteration
        totalbatches: Number of batches within loader
    """
    def __init__(self, root_dir: str, batch_size=256, seed=42):
        """
        Initialise TrainTestLoader.

        Keyword arguments:
            root_dir: Filepath where dataset is stored
        
        Optional keyword arguments:
            batch_size: Number of data samples processed in a single iteration
        """
        
        super().__init__(root_dir, seed)
        self.__loader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
            )


        # Batch size reference for other programs
        self.__batch_size = batch_size

        self.__totalbatches = len(self.__loader)


    @property
    def loader(self):
        return self.__loader

    @property
    def totalbatches(self):
        return self.__totalbatches



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

    print(f"Amount of items in dataset: {len(ttl.__dataset)}")
    dataiter = iter(ttl.__trainloader)
    images, labels = next(dataiter)
    imshow(make_grid(images))
    print(" ".join(f"{labels[j].item():5}" for j in range(ttl.__batch_size)))