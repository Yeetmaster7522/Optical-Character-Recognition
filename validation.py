from torch import load, device, accelerator, no_grad, amp, autocast, backends, autograd, float16, compile, set_float32_matmul_precision
from torch import max as tmax
from torch.nn import Module, functional
import pandas as pd
from tqdm import tqdm

from dataloader import CHARSET



class Validator:
    """
    Automates prediction/validation of machine learning models

    Attributes:
        path: Filepath where model is saved
        model: Which model architecture to use
        root_dir: Filepath where test images are
        device: Which device will be used for calculations during prediction
    """
    
    def __init__(self, path: str, model: Module):
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


        # Finds hardware accelerators and utilises that if possible. (CUDA, ROCm, TPU, MPS)
        self.device = accelerator.current_accelerator().type if accelerator.is_available() else 'cpu'
        print(f"Using device: {self.device}")



    def update(self, model, path):
        self.model = model
        self.path = path



    def check_acc(self, fl) -> dict:
        """
        Runs model in prediction mode though data in self.root_dir
        """

        d = device(self.device)


        # Enable NVIDIA cuDNN auto tuner
        backends.cudnn.benchmark = True


        # Disable debugging APIs
        autograd.set_detect_anomaly(False)
        autograd.profiler.profile(False)


        set_float32_matmul_precision("high")


        # Creates model object and puts it on self.device
        # Then loads the state_dict from saved models
        net = self.model().to(d)
        net.load_state_dict(load(self.path, weights_only=True))
        compiled_net = compile(net, mode="max-autotune")


        # Load testloader
        tl = fl.loader


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
        compiled_net.eval()

        with tqdm(
                total=fl.totalbatches,
                desc=f"Testing progress"
            ) as pbar:
            with no_grad():
                # Iterates through inputs and labels in testloader
                for i, (images, labels) in enumerate(tl):
                    pbar.update(1)

                    # Puts inputs and labels on self.device
                    images, labels = images.to(d), labels.to(d)

                    with autocast(device_type=self.device, dtype=float16):
                        # Get top prediction and confidence
                        outputs = compiled_net(images)

                        probs = functional.softmax(outputs, dim=1)
                        conf, predicted = tmax(probs, 1)  # https://stackoverflow.com/questions/69154022/how-to-get-confidence-score-from-a-trained-pytorch-model

                    # Iterates through given images.
                    for j in range(len(images)):
                        # Gets index of image in the original dataset and uses it to get filename
                        filename = tl.dataset.images[i * tl.batch_size + j]["filename"]


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
    import models
    from dataloader import FullLoader

    validator = Validator(
        path="models/models_F/ocr_v5_t0.052_v0.044.pt",
        model=models.ocr_v5
    )
    print("starting validation")
    results = validator.check_acc(FullLoader("dataset/character_images_no_noise"))
    validator.save_to_csv(results, "results.csv")