from torch import load, device, accelerator, no_grad
from torch import max as tmax

from models import ocr_v4 as model
from dataloader import testloader as tl

PATH = "models_T/model_4_T_27.pt"
DEVICE = device(accelerator.current_accelerator().type if accelerator.is_available() else 'cpu')

if __name__ == "__main__":
    net = model().to(DEVICE)
    net.load_state_dict(load(PATH, weights_only=True))

    print(f"Using device: {DEVICE}")

    correct = 0
    total = 0

    with no_grad():
        print("Validation started")

        for images, labels in tl:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = net(images)
            
            _, predicted = tmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy [{total}]: {100 * correct // total}%")


"""
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
"""