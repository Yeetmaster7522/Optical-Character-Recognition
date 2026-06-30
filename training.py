import torch.optim as optim
from torch import nn, save, device, no_grad, amp, autocast, backends, autograd, float16, compile, set_float32_matmul_precision

from tqdm import tqdm

from dataloader import TrainTestLoader



class Trainer:
    """
    Automates machine learning training process.

    Attributes:
        name: What the model weights will be saved as on hard drive
        model: What model architecture will be trained
        save_folder: Filepath where model weights will be saved
        device: What device to use for computation
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
            model: nn.Module,
            save_folder: str,
            device: str,
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

            max_epochs (optional): Epoch at which training will always stop at (if early stopping not achieved)
            lr (optional): Learning rate used for training
            tloss_checkpoint (optional): Checkpoint for training loss at which model weights will start being
            saved
            max_patience (optional): How long to wait for model training loss to drop before stopping training
        """

        # Setting class attributes
        self.__name = name
        self.__model = model
        self.__save_folder = save_folder
        self.__max_epochs = max_epochs
        self.__lr = lr
        self.__tloss_checkpoint = tloss_checkpoint
        self.__max_patience = max_patience
        self.__device = device



    def update(self, name, model, save_folder):
        self.__name = name
        self.__model = model
        self.__save_folder = save_folder



    def save_model(self, state_dict: dict, filepath: str):
        """
        Saves model

        Keyword arguments:
            state_dict: Models current state. Obtainable via .state_dict().
            filepath: Where model will be saved.
        """
        
        save(state_dict, filepath)



    def train(self, ttl: TrainTestLoader):
        """
        Training loop. Returns train loss and validation loss.

        Keyword arguments:
            ttl: TrainTestLoader class
        """

        d = device(self.__device)


        # Enable NVIDIA cuDNN auto tuner
        backends.cudnn.benchmark = True


        # Disable debugging APIs
        autograd.set_detect_anomaly(False)
        autograd.profiler.profile(False)


        set_float32_matmul_precision("high")


        # Creates and compiles model object
        net = self.__model().to(d)
        
        # Leverage Triton or template based matrix multiplications with CUDA graphs
        compiled_net = compile(net, mode="max-autotune")


        # States criteria to evaluate loss and optimiser
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(compiled_net.parameters(), lr=self.__lr)


        # Load train and test loader
        trainloader = ttl.trainloader
        testloader = ttl.testloader


        # Train loss and validation loss over epochs
        tlosses = []
        vlosses = []

        lowest_loss = self.__tloss_checkpoint # Lowest loss achieved
        patience_counter = 0 # Epochs gone without achieving a new lowest loss


        # Create GradScaler once at beginning of training
        scaler = amp.GradScaler(self.__device)

        # Start training
        for epoch in range(self.__max_epochs):
            running_tloss = 0
            running_vloss = 0

            with tqdm(
                total=ttl.train_totalbatches+ttl.test_totalbatches, 
                desc=f"Epoch: {epoch+1}"
                ) as pbar:
                # Set model into training mode
                compiled_net.train()

                # Iterates through inputs and labels in trainloader
                for inputs, labels in trainloader:
                    pbar.update(1)

                    inputs, labels = inputs.to(d), labels.to(d)

                    # Clears accumulated gradients
                    optimizer.zero_grad(set_to_none=True)

                    # Runs forward pass with autocasting
                    with autocast(device_type=self.__device, dtype=float16):
                        outputs = compiled_net(inputs)
                        loss = criterion(outputs, labels)

                    # Add to running loss
                    running_tloss += loss.item()

                    # Scales loss and then calls backward on scaled loss to create scaled gradients
                    scaler.scale(loss).backward()
                    
                    # First unscales gradients and then update model parameters
                    scaler.step(optimizer)

                    # Updates scale for next iteration
                    scaler.update()


                # Set model to evaluation mode
                compiled_net.eval()

                # Disable gradient calculation
                with no_grad():
                    # Iterates through inputs and labels in testloader
                    for vinputs, vlabels in testloader:
                        pbar.update(1)

                        vinputs, vlabels = vinputs.to(d), vlabels.to(d)

                        with autocast(device_type=self.__device, dtype=float16):
                            # Get outputs and calculate validation loss
                            voutputs = compiled_net(vinputs)
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
            print(f"\t|-> Train loss: {tloss:.3f}\n\t|-> Validation loss: {vloss:.3f}")


            # If validation loss avg over epoch smaller than lowest loss then will save
            # Else, adds to patience counter
            if vloss < lowest_loss:
                lowest_loss = vloss
                patience_counter = 0

                self.save_model(net.state_dict(), f"{self.__save_folder}/{self.__name}_t{tloss:.3f}_v{vloss:.3f}.pt")
                print("\t|-> Model saved")
            else:
                patience_counter += 1


            # If ran out of patience then will stop training
            # Else outputs how much patience it has used up
            if patience_counter > self.__max_patience:
                print("\t|-> EARLY STOPPING TRIGGERED")
                break
            else:
                print(f"\t|-> Patience: {patience_counter}/{self.__max_patience}")


        return tlosses, vlosses



if __name__ == "__main__":
    from models import ocr_v1 as model

    trainer = Trainer(
        name=model.__name__,
        model=model,
        save_folder="models/models_F",
        device="cpu"
    )
    # Train model, tloss is train loss, and vloss is validation/test loss
    tloss, vloss = trainer.train()

    # Finished training
    print("Training Finished!")