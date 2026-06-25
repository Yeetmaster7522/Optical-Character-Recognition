from torch import load, device, accelerator, no_grad
from torch import max as tmax
from torch.nn.functional import softmax
from torch.nn import Module
import pandas as pd

from dataloader import TrainTestLoader, CHARSET

class Validator:
    """
    Automates prediction/validation of machine learning models

    Attributes:
        path: Filepath where model is saved
        model: Which model architecture to use
        root_dir: Filepath where test images are
        device: Which device will be used for calculations during prediction
    """
    
    def __init__(self, path: str, model: Module, root_dir: str):
        """
        Initialises Validator.

        Keyword arguments:
            path: Filepath where model is saved
            model: Which model architecture to use
            root_dir: Filepath where test images are
        """
        
        # Set class attributes
        self.path = path
        self.model = model
        self.root_dir = root_dir

        # Finds hardware accelerators and utilises that if possible. (CUDA, ROCm, TPU, MPS)
        self.device = device(accelerator.current_accelerator().type if accelerator.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def check_acc(self) -> dict:
        """
        Runs model in prediction mode though data in self.root_dir
        """
        
        # Creates model object and puts it on self.device
        # Then loads the state_dict from saved models
        net = self.model().to(self.device)
        net.load_state_dict(load(self.path, weights_only=True))

        # Load testloader
        ttl = TrainTestLoader(root_dir=self.root_dir, split=0.99)
        tl = ttl.testloader
        
        # Courtesy of Copilot. 
        subset = tl.dataset
        og_dataset = subset.dataset
        indices = subset.indices

        # Amount predicted correctly and total amount of data
        correct = 0
        total = 0

        # dictionary of results
        results = {
            "filename": [],
            "actual letter": [],
            "predicted letter": [],
            "result": [],
            "confidence score": []
        }

        # Set mode in evaluation mode and disable gradient calculation
        net.eval()

        with no_grad():
            # Iterates through inputs and labels in testloader
            for i, (images, labels) in enumerate(tl):
                # Puts inputs and labels on self.device
                images, labels = images.to(self.device), labels.to(self.device)

                # Get top prediction and confidence
                outputs = net(images)
                probs = softmax(outputs, dim=1)
                conf, predicted = tmax(probs, 1)  # https://stackoverflow.com/questions/69154022/how-to-get-confidence-score-from-a-trained-pytorch-model

                # Iterates through given images.
                for j in range(len(images)):
                    # Help from Copilot.
                    # Gets index of image in the original dataset and uses it to get filename
                    og_idx = indices[i * tl.batch_size + j]
                    filename = og_dataset.images[og_idx]["filename"]

                    # Gets predicted and true item
                    pred = predicted[j].item()
                    true = labels[j].item()

                    # Saves results
                    results["filename"].append(filename)
                    results["actual letter"].append(CHARSET[true])
                    results["predicted letter"].append(CHARSET[pred])
                    results["result"].append("Pass" if pred == true else "Fail")
                    results["confidence score"].append(f"{conf[j]:.2f}")

                    # Increments correct and total counter
                    total += 1
                    correct += (pred == true)

        # Display accuracy in 2 d.p.
        print(f"Accuracy [{total}]: {100 * correct / total:.2f}%")

        return results

    def save_to_csv(self, table: dict, filepath: str):
        """
        Saves table to filepath as a csv file.

        Keyword arguments:
            table: A dictionary of keys with arrays as items
            filepath: Filepath where csv is stored
        """
        
        # Converts table into DataFrame object and saves it as csv
        df = pd.DataFrame(table)
        df.to_csv(filepath, index=False)

if __name__ == "__main__":
    from models import ocr_v2 as model

    validator = Validator(
        path="models/models_F/ocr_v2_F_5.pt",
        model=model,
        root_dir="dataset/character_images_no_noise"
    )
    print("starting validation")
    results = validator.check_acc()
    validator.save_to_csv(results, "results.csv")