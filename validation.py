from torch import load, device, accelerator, no_grad
from torch import max as tmax

from models import ocr_v1 as model
from dataloader import TrainTestLoader, CHARSET

class Validator:
    def __init__(self, path, model, root_dir):
        self.path = path
        self.model = model
        self.root_dir = root_dir

        self.device = device(accelerator.current_accelerator().type if accelerator.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def check_accuracy(self):
        net = self.model().to(self.device)
        net.load_state_dict(load(self.path, weights_only=True))

        tl = TrainTestLoader(root_dir=self.root_dir).trainloader

        correct = 0
        total = 0

        class_correct = {classname: 0 for classname in CHARSET}
        class_total = {classname: 0 for classname in CHARSET}

        with no_grad():
            print("Validation started")

            for images, labels in tl:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = net(images)
                _, predicted = tmax(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        class_correct[CHARSET[int(label)]] += 1
                    class_total[CHARSET[int(label)]] += 1

        print(f"Accuracy [{total}]: {100 * correct / total:.3f}%")

        for classname, correct_count in class_correct.items():
            accuracy = 100 * float(correct_count) / class_total[classname]
            print(f"Accuracy for class: {classname} is {accuracy:.3f}%")

if __name__ == "__main__":
    validator = Validator(
        path="models/models_F/model_1_F_3.pt",
        model=model,
        root_dir="dataset/character_images_no_noise"
    )
    validator.check_accuracy()