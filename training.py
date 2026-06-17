import torch.optim as optim
from torch import nn, save, device, accelerator

import matplotlib.pyplot as plt

from models import ocr_v1 as model
from dataloader import TrainTestLoader as ttl

class Trainer:
    def __init__(
            self,
            name,
            model:nn.Module,
            save_folder,
            root_dir,
            max_epochs=100,
            lr=1e-3,
            training_loss_save_point=0.18,
            max_patience=7
            ):
        self.name = name
        self.model = model
        self.save_folder = save_folder
        self.root_dir = root_dir
        self.max_epochs = max_epochs
        self.lr = lr
        self.training_loss_save_point = training_loss_save_point
        self.max_patience = max_patience

        self.device = device(accelerator.current_accelerator().type if accelerator.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def save_model(self, state_dict, name):
        save(state_dict, name)
        print("Model saved")


    def train(self):
        net = self.model().to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(net.parameters(), lr=self.lr)

        tl = ttl(root_dir=self.root_dir).trainloader

        losses = []
        lowest_loss = self.training_loss_save_point
        patience_counter = 0

        for epoch in range(self.max_epochs):
            running_loss = 0.0
            running_losses = []
    
            for i, (inputs, labels) in enumerate(tl, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

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

                        self.save_model(net.state_dict(), f"{self.save_folder}/{self.name}_{epoch+1}.pt")

                    running_losses.append(running_loss/200)
                    running_loss = 0.0

            losses.append(min(running_losses))

            patience_counter += 1

            if patience_counter > self.max_patience:
                print("EARLY STOPPING TRIGGERED")
                break
            else:
                print(f"Patience: {patience_counter}/{self.max_patience}")

        return losses

    def plot(self, values):
        plt.plot([e+1 for e in range(len(values))], values)
        plt.show()

if __name__ == "__main__":
    trainer = Trainer(
        name="model_1_F",
        model=model,
        save_folder="models/models_F",
        root_dir="dataset/character_images_no_noise",
        max_epochs=3
    )
    losses = trainer.train()
    print("Training Finished!")
    trainer.plot(losses)