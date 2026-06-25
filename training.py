import torch.optim as optim
from torch import nn, save, accelerator, no_grad, amp, autocast, backends, bfloat16

import matplotlib.pyplot as plt

from dataloader import TrainTestLoader as TTL

class Trainer:
    """
    Automates machine learning training process.

    Attributes:
        name: What the model weights will be saved as on hard drive
        model: What model architecture will be trained
        save_folder: Filepath where model weights will be saved
        root_dir: Filepath to access training data
        max_epochs: Epoch at which training will always stop at (if early stopping not achieved)
        lr: Learning rate used for training
        tloss_checkpoint: Checkpoint for training loss at which model weights will start being
        saved
        max_patience: How long to wait for model training loss to drop before stopping training
        device: Which device will be used for calculations during training
    """
    
    def __init__(
            self,
            name: str,
            model:nn.Module,
            save_folder: str,
            root_dir: str,
            max_epochs=100,
            lr=2e-3,
            tloss_checkpoint=0.18,
            max_patience=7
            ):
        """
        Initialises Trainer.

        Keyword arguments:
            name: What the model weights will be saved as on hard drive
            model: What model architecture will be trained
            save_folder: Filepath where model weights will be saved
            root_dir: Filepath to access training data
            max_epochs: Epoch at which training will always stop at (if early stopping not achieved)
            lr: Learning rate used for training
            tloss_checkpoint: Checkpoint for training loss at which model weights will start being
            saved
            max_patience: How long to wait for model training loss to drop before stopping training
        """

        # Setting class attributes
        self.name = name
        self.model = model
        self.save_folder = save_folder
        self.root_dir = root_dir
        self.max_epochs = max_epochs
        self.lr = lr
        self.tloss_checkpoint = tloss_checkpoint
        self.max_patience = max_patience

        # Finds hardware accelerators and utilises that if possible. (CUDA, ROCm, TPU, MPS)
        self.device = accelerator.current_accelerator().type if accelerator.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def save_model(self, state_dict: dict, filepath: str):
        """
        Saves model

        Keyword arguments:
            state_dict: Models current state. Obtainable via .state_dict().
            filepath: Where model will be saved.
        """
        
        save(state_dict, filepath)
        print("Model saved")


    def train(self):
        """
        Training loop.
        """

        backends.cudnn.benchmark = True
        
        # Creates model object and puts it on self.device
        net = self.model().to(self.device)

        # States criteria to evaluate loss and optimiser
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(net.parameters(), lr=self.lr)

        # Load train and test loader
        ttl = TTL(root_dir=self.root_dir)
        trainloader = ttl.trainloader
        testloader = ttl.testloader

        # Train loss and validation loss over epochs
        tlosses = []
        vlosses = []

        lowest_loss = self.tloss_checkpoint # Lowest loss achieved
        patience_counter = 0 # Epochs gone without achieving a new lowest loss

        # Create GradScaler once at beginning of training
        scaler = amp.GradScaler(self.device)

        # Start training
        for epoch in range(self.max_epochs):
            # Set model into training mode and reset training running loss
            net.train()
            running_tloss = 0

            # Iterates through inputs and labels in trainloader
            for i, (inputs, labels) in enumerate(trainloader, 0):
                # Clears accumulated gradients
                optimizer.zero_grad(set_to_none=True)

                # Runs forward pass with autocasting
                with autocast(device_type=self.device, dtype=bfloat16):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                # Add to running loss
                running_tloss += loss.item()

                # Scales loss and then calls backward on scaled loss to create scaled gradients
                scaler.scale(loss).backward()
                
                # First unscales gradients and then update model parameters
                scaler.step(optimizer)

                # Updates scale for next iteration
                scaler.update()

                # Print progress every 100 mini-batches
                if i % 100 == 99:
                    print(f'[{epoch + 1}, {i + 1:5d}]')

            # Set model to evaluation mode and set running validation loss
            net.eval()
            running_vloss = 0

            # Disable gradient calculation
            with no_grad():
                # Iterates through inputs and labels in testloader
                for vinputs, vlabels in testloader:
                    with autocast(device_type=self.device, dtype=bfloat16):
                        # Get outputs and calculate validation loss
                        voutputs = net(vinputs)
                        vloss = criterion(voutputs, vlabels)

                    # Add validation loss to running validation loss
                    running_vloss += vloss.item()

            # Calculate average training loss and validation loss
            tloss = running_tloss / ttl.train_totalbatches
            vloss = running_vloss / ttl.test_totalbatches

            # Save averages in array
            tlosses.append(tloss)
            vlosses.append(vloss)

            # Display current train and validation loss
            print(f'loss: {tloss:.3f}\nvloss: {vloss:.3f}')

            # If validation loss avg over epoch smaller than lowest loss then will save
            # Else, adds to patience counter
            if vloss < lowest_loss:
                lowest_loss = vloss
                patience_counter = 0
                self.save_model(net.state_dict(), f"{self.save_folder}/{self.name}_t{tloss:.3f}_v{vloss:.3f}.pt")
            else:
                patience_counter += 1

            # If ran out of patience then will stop training
            # Else outputs how much patience it has used up
            if patience_counter > self.max_patience:
                print("EARLY STOPPING TRIGGERED")
                break
            else:
                print(f"Patience: {patience_counter}/{self.max_patience}")

        return tlosses, vlosses

if __name__ == "__main__":
    from models import ocr_v1 as model

    trainer = Trainer(
        name=model.__name__,
        model=model,
        save_folder="models/models_F",
        root_dir="dataset/character_images_no_noise",
        max_epochs=10
    )
    # Train model, tloss is train loss, and vloss is validation/test loss
    tloss, vloss = trainer.train()

    # Calculate difference between tloss and vloss over epochs
    loss_dif = [abs(t-vloss[i]) for i,t in enumerate(tloss)]

    # Finished training and creating graphs
    print("Training Finished!")

    # Creates an array of epoch numbers from 1 to n
    epochs = [e+1 for e in range(len(tloss))]

    # Plot training and validation loss
    plt.plot(epochs, tloss, color="red", label="Train loss")
    plt.plot(epochs, vloss, color="green", label="Test loss")
    
    # Plot loss difference between tloss and vloss
    plt.plot(epochs, loss_dif, color="blue", linestyle="-.", label="Loss diff")
    
    # Show graph
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()