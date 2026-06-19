from torch import load, device, accelerator, no_grad
from torch import max as tmax
from torch.nn.functional import softmax
import pandas as pd

from dataloader import TrainTestLoader, CHARSET

class Validator:
    def __init__(self, path, model, root_dir):
        self.path = path
        self.model = model
        self.root_dir = root_dir

        self.device = device(accelerator.current_accelerator().type if accelerator.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def check_acc(self):
        net = self.model().to(self.device)
        net.load_state_dict(load(self.path, weights_only=True))

        ttl = TrainTestLoader(root_dir=self.root_dir)
        tl = ttl.trainloader
        
        subset = tl.dataset
        og_dataset = subset.dataset
        indices = subset.indices

        correct = 0
        total = 0

        results = {
            "filename": [],
            "predicted letter": [],
            "result": [],
            "confidence score": []
        }

        with no_grad():
            for i, (images, labels) in enumerate(tl):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = net(images)
                probs = softmax(outputs, dim=1)
                conf, predicted = tmax(probs, 1)  # https://stackoverflow.com/questions/69154022/how-to-get-confidence-score-from-a-trained-pytorch-model

                for j in range(len(images)):
                    og_idx = indices[i * tl.batch_size + j]
                    filename = og_dataset.images[og_idx]["filename"]

                    pred = predicted[j].item()
                    true = labels[j].item()

                    results["filename"].append(filename)
                    results["predicted letter"].append(pred)
                    results["result"].append(pred == true)
                    results["confidence score"].append(f"{conf[j]:.2f}")

                    total += 1
                    correct += (pred == true)

        print(f"Accuracy [{total}]: {100 * correct / total:.2f}%")
        return results

    def save_to_csv(self, table, filepath):
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