from torch import load, device, accelerator, no_grad
from torch import max as tmax
import pandas as pd

from dataloader import TrainTestLoader, CHARSET

class Validator:
    def __init__(self, path, model, root_dir):
        self.path = path
        self.model = model
        self.root_dir = root_dir

        self.device = device(accelerator.current_accelerator().type if accelerator.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def check_overall_acc(self):
        net = self.model().to(self.device)
        net.load_state_dict(load(self.path, weights_only=True))

        tl = TrainTestLoader(root_dir=self.root_dir).trainloader

        correct = 0
        total = 0

        with no_grad():
            for images, labels in tl:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = net(images)
                _, predicted = tmax(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy [{total}]: {100 * correct / total:.3f}%")

    def check_class_acc(self):
        net = self.model().to(self.device)
        net.load_state_dict(load(self.path, weights_only=True))

        tl = TrainTestLoader(root_dir=self.root_dir).trainloader

        class_correct = {classname: 0 for classname in CHARSET}
        class_total = {classname: 0 for classname in CHARSET}

        with no_grad():
            for images, labels in tl:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = net(images)
                _, predicted = tmax(outputs, 1)

                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        class_correct[CHARSET[int(label)]] += 1
                    class_total[CHARSET[int(label)]] += 1

        for classname, correct_count in class_correct.items():
            accuracy = 100 * float(correct_count) / class_total[classname]
            print(f"Accuracy for class: {classname} is {accuracy:.3f}%")

    def to_df(self, **kwargs):
        """
        File name (name, not path)
        predicted letter
        result (Pass, Fail)
        confidence score (2 d.p.)
        """

        df = pd.DataFrame(kwargs)
        return df

if __name__ == "__main__":
    from models import ocr_v2 as model

    validator = Validator(
        path="models/models_F/ocr_v2_F_5.pt",
        model=model,
        root_dir="dataset/character_images_no_noise"
    )
    validator.check_overall_acc()