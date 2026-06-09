import torch.optim as optim
from torch import nn, save, device, accelerator

import matplotlib.pyplot as plt

from models import ocr_v4 as model
from dataloader import trainloader as tl

NAME = "model_4_T"
SAVE_FOLDER = "models_T" #F=No noise, T=Noise
EPOCHS = 100
LR = 1e-3
TRAINING_LOSS_SAVE_POINT = 0.18
MAX_PATIENCE = 7

DEVICE = device(accelerator.current_accelerator().type if accelerator.is_available() else 'cpu')

if __name__ == "__main__":
    net = model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.AdamW(net.parameters(), lr=LR)
    trainloader = tl

    losses = []
    lowest_loss = TRAINING_LOSS_SAVE_POINT
    patience_counter = 0

    print(f"Using device: {DEVICE}")

    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_losses = []
 
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                if running_loss / 200 < lowest_loss:
                    lowest_loss = running_loss / 200
                    patience_counter = 0

                    save(net.state_dict(), f"{SAVE_FOLDER}/{NAME}_{epoch+1}.pt")
                    print("Model saved")

                running_losses.append(running_loss/200)
                running_loss = 0.0

        losses.append(min(running_losses))

        patience_counter += 1

        if patience_counter > MAX_PATIENCE:
            print("EARLY STOPPING TRIGGERED")
            break
        else:
            print(f"Patience: {patience_counter}/{MAX_PATIENCE}")

    print('Finished Training')

    plt.plot([e+1 for e in range(len(losses))], losses)
    plt.show()