import os
import matplotlib.pyplot as plt
from torch.nn import Module
from sklearn.metrics import ConfusionMatrixDisplay #https://stackoverflow.com/questions/74020233/how-to-plot-confusion-matrix-in-pytorch

import models
from training import Trainer
from validation import Validator
from dataloader import CHARSET, char_to_idx

class Main:
    """
    Streamlines the entire ML pipeline for a user including training and testing.

    Attributes:
        model_classes: Array of nn.Modules
        model_dir: Filepath where models are saved
        train_dir Filepath where training images are saved
        test_dir: Filepath where test images are saved
    """
    
    def __init__(self, model_dir="", train_dir="", test_dir=""):
        """
        Initialise Main.
        
        Keyword arguments:
            model_dir: Filepath where models are saved
            train_dir: Filepath where training images are saved
            test_dir: Filepath where test images are saved
        """
        
        print("WELCOME")
        
        self.model_classes = [
            models.ocr_v1,
            models.ocr_v2,
            models.ocr_v3,
            models.ocr_v4,
            models.ocr_v5
        ]
        self.model_dir = model_dir
        self.train_dir = train_dir
        self.test_dir = test_dir

    def list_models(self):
        """
        Lists every nn.Module class within self.model_classes in the terminal
        in the format:
            [i]: Name of class
        """
        
        for i in range(len(self.model_classes)):
            print(f"[{i}]: {self.model_classes[i].__name__}")

    def loop(self):
        """
        Mainloop for user interaction.

        Asks user whether they want to train, predict, or exit the program.
        """
        
        while True:
            # Get user input
            user_inp = self.get_inp(
                display="\nWhat would you like to do?\n(T for train, P for predict, E for exit)\n",
                expected=["t", "p", "e"]
                )

            # If user wants to exit
            if user_inp == "e":
                break

            # If user wants to train/predict
            self.select_train_pred(user_inp)

    def get_inp(self, display: str, expected=[]):
        """
        Cleanses and validates user input.

        Keyword arguments:
            display: What will be printed to the console
            expected (optional): User input must be one of the values listed here
        """
        
        while True:
            # Get user input and remove whitespace
            user_inp = input(display).strip()

            # Try turning user input to int.
            # If failed, then will make it lowercase.
            try:
                user_inp = int(user_inp)
            except ValueError:
                user_inp = user_inp.lower()

            # Returns user input if it is in expected or
            # returns user input if there is no expected values.
            if expected == [] or user_inp in expected:
                return user_inp
            else:
                print("Invalid input")

    def select_train_pred(self, choice: str):
        """
        Selects training or prediction mode based on keyword argument: choice
        """
        
        # Asks the user where to get and save models if haven't already
        if self.model_dir == "":
            self.model_dir = self.get_inp("Where do I get and save models? ")

        # Training mode
        if choice == "t":
            print("\nTraining mode chosen")

            # Asks user where training data is if haven't already
            if self.train_dir == "":
                self.train_dir = self.get_inp("Where do I get training data? ")

            # Lists model architectures and asks which one they would like to train
            self.list_models()
            model_idx = self.get_inp(
                display="Select model type from above: ",
                expected=[i for i in range(len(self.model_classes))]
                )
            model = self.model_classes[model_idx]
            print(f"Model will be saved as: {model.__name__}\nIn folder: {self.model_dir}")

            # Train model
            self.train(model)

        # Prediction mode
        elif choice == "p":
            print("\nPrediction mode chosen")

            # Asks users where training data is if haven't already
            if self.test_dir == "":
                self.test_dir = self.get_inp("Where do I get testing data? ")

            # Lists model architectures and asks which one they would like to train
            self.list_models()
            model_idx = self.get_inp(
                display="Select model type from above: ",
                expected=[i for i in range(len(self.model_classes))]
                )
            model = self.model_classes[model_idx]
            
            # Lists saved models based off chosen model architecture and asks which one
            # they would like to train
            files = [f for f in os.listdir(self.model_dir) if model.__name__ in f]
            for i in range(len(files)):
                print(f"[{i}]: {files[i]}")

            saved_idx = self.get_inp(
                display="Select saved model from above: ",
                expected=[i for i in range(len(files))]
            )

            self.test(model, files[saved_idx]) # Predict images using model

    def train(self, model: Module):
        """
        Trains model and outputs final accuracy and training/testing loss over epochs

        Keyword argument:
            model: Model architecture that will be trained
        """

        # Create instance of Trainer class
        trainer = Trainer(
            name=model.__name__,
            model=model,
            save_folder=self.model_dir,
            root_dir=self.train_dir
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
        
    def test(self, model: Module, filename: str):
        """
        Puts model into prediction mode and outputs accuracy over test dataset.

        Keyword arguments:
            model: Model architecture that will be used for prediction
            filename: Filename of the model weights that will be loaded into model architecture
        """
        
        # Create instance of Validator class
        validator = Validator(
            path=f"{self.model_dir}/{filename}",
            model=model,
            root_dir=self.test_dir
        )

        print("Starting test...")

        # Run model through images in self.test_dir and gets results
        results = validator.check_acc()

        # Save results to CSV
        validator.save_to_csv(results, "results.csv")
        print("Results saved in results.csv")

        # Show results in graphical form
        self.show_test_results(results)

    def show_test_results(self, results: dict):
        """
        Turns data from results into a confusion matrix and bar charts for confidence and
        accuracy per character.

        Keyword arguments:
            results: Results from running model through test images. 
            Is in format:
                {
                    "filename": [],
                    "actual letter": [],
                    "predicted letter": [],
                    "result": [],
                    "confidence score": []
                }
        """

        # Starting values
        y_test = []
        y_pred = []
        confidence_sum = [0 for _ in CHARSET]
        correct = [0 for _ in CHARSET]
        totals = [0 for _ in CHARSET]

        # Iterates through each entry in results and notes down prediction, and the actual
        # character as well as incrementing correct and totals as well as adding to sum of
        # confidence per character.
        for i in range(len(results["filename"])):
            # filename = results["filename"][i]
            pred = results["predicted letter"][i] # Precicted letter
            actual = results["actual letter"][i] # Actual letter
            result = results["result"][i] # Result of prediction "Pass" or "Fail"
            conf = float(results["confidence score"][i]) # Confidence score
            j = CHARSET.index(actual) # Index of actual letter within CHARSET

            # Character/letter has to be turned into index here to match neural network output
            y_test.append(char_to_idx[actual]) # Appends actual letter to y_test
            y_pred.append(char_to_idx[pred]) # Appends predicted letter to y_pred

            # Adds confidence to sum at index j
            confidence_sum[j] += float(conf)

            # If the result was a pass then it will increment correct at index j
            if result == "Pass":
                correct[j] += 1

            # Adds total at index j
            totals[j] += 1

        # Calculates confidence avg and percentage of correct predictions up to 2 d.p. and puts
        # them into a dictionary for plotting
        char_means = {
            "confidence_avg": [round(conf/totals[i], 2) for i,conf in enumerate(confidence_sum)],
            "correct_pct": [round(count/totals[i], 2) for i,count in enumerate(correct)]
        }

        # Creates confusion matrix based of y_test and y_pred
        ConfusionMatrixDisplay.from_predictions(
            y_test, 
            y_pred,
            include_values=False,
            display_labels=CHARSET,
            normalize="true",
            cmap="Blues"
        )
        # Show confusion matrix
        plt.show()

        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
        # Create grouped barchart
        fig, ax = plt.subplots(layout="constrained")

        res = ax.grouped_bar(char_means, tick_labels=CHARSET, group_spacing=1)
        for container in res.bar_containers:
            ax.bar_label(container, padding=3)

        # Set labels and show barchart
        ax.set_ylabel("%")
        ax.set_title("Confidence and Accuracy by Character")
        ax.legend(loc="upper left", ncols=2)
        plt.show()



if __name__ == "__main__":
    main = Main(
        model_dir="new_models/models_T", 
        train_dir="dataset/character_images_with_noise",
        test_dir="dataset/character_images_with_noise"
    )
    main.loop()