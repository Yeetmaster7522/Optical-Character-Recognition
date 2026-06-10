from torch import load, device, accelerator, no_grad
from torch import max as tmax

from models import ocr_v4 as model
from dataloader import testloader as tl
from dataloader import char_to_idx as cti

PATH = "models_T/model_4_T_27.pt"
DEVICE = device(accelerator.current_accelerator().type if accelerator.is_available() else 'cpu')

if __name__ == "__main__":
    net = model().to(DEVICE)
    net.load_state_dict(load(PATH, weights_only=True))

    print(f"Using device: {DEVICE}")

    correct = 0
    total = 0

    class_correct = {classname: 0 for classname in cti}
    class_total = {classname: 0 for classname in cti}

    with no_grad():
        print("Validation started")

        for images, labels in tl:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = net(images)
            _, predicted = tmax(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    class_correct[cti[label]] += 1
                class_total[cti[label]] += 1

    print(f"Accuracy [{total}]: {100 * correct / total:.3f}%")

    for classname, correct_count in class_correct.items():
        accuracy = 100 * float(correct_count) / class_total[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")