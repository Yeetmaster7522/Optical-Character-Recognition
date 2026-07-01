from torch import nn, accelerator

import os

import models
from training import Trainer
from validation import Validator
from dataloader import TrainTestLoader, FullLoader, CHARSET, char_to_idx, CHARSET_W
import grapher



class Main:
    """
    Streamlines the entire ML pipeline for a user including training and testing.
    """
    
    def __init__(self, model_dir="", train_dir="", test_dir=""):
        """
        Initialise Main.
        
        Keyword arguments:
            model_dir: Filepath where models are saved
            train_dir: Filepath where training images are saved
            test_dir: Filepath where test images are saved
        """

        self.__model_classes: list[nn.Module] = [
            models.ocr_v1,
            models.ocr_v2,
            models.ocr_v3,
            models.ocr_v4,
            models.ocr_v5,
            models.ocr_wingding
        ]
        self.__model_dir: str = model_dir
        self.__train_dir: str = train_dir
        self.__test_dir: str = test_dir


        self.__ttl: TrainTestLoader = None
        self.__fl: FullLoader = None

        # Finds hardware accelerators and utilises that if possible. (CUDA, ROCm, TPU, MPS)
        device = accelerator.current_accelerator().type if accelerator.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        self.__trainer: Trainer = Trainer(
            name="",
            model=None,
            save_folder=self.__model_dir,
            device=device
        )
        self.__validator: Validator = Validator(
            path="",
            model=None,
            device=device
        )

        print("WELCOME")



    def list_models(self):
        """
        Lists every nn.Module class within self.model_classes in the terminal
        in the format:
            [i]: Name of class
        """
        
        for i in range(len(self.__model_classes)):
            print(f"[{i}]: {self.__model_classes[i].__name__}")



    def loop(self):
        """
        Mainloop for user interaction.

        Asks user whether they want to train, predict, or exit the program.
        """
        
        # Get user input
        user_inp: str = self.get_inp(
            display="\nWhat would you like to do?\n(T for train, P for predict, E for exit)\n",
            expected=["t", "p", "e"]
            )

        # If user wants to train/predict
        if user_inp != "e":
            self.select_train_pred(user_inp)
            self.loop()
        # If user wants to exit then the loop() method will not be called again



    def get_inp(self, display: str, expected=[], is_filepath=False) -> str | int:
        """
        Cleanses and validates user input.

        Keyword arguments:
            display: What will be printed to the console
            expected (optional): User input must be one of the values listed here
            is_filepath (optional): If true then it will ensure that user input is a filepath
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


            is_valid: bool = False

            # Returns user input if it is in expected or
            # returns user input if there is no expected values.
            if expected == [] and not is_filepath or user_inp in expected:
                is_valid = True
            elif is_filepath and os.path.isdir(user_inp): # check if filepath exists
                is_valid = True
                user_inp = os.path.normpath(user_inp) # Clean up filepath in case it's messy
            else:
                print("Invalid input")


            if is_valid:
                return user_inp



    def select_train_pred(self, choice: str):
        """
        Selects training or prediction mode based on keyword argument: choice
        """
        
        # Asks the user where to get and save models if haven't already
        if self.__model_dir == "":
            self.__model_dir = self.get_inp("Where do I get and save models? ")


        # Training mode
        if choice == "t":
            print("\nTraining mode chosen")

            # Asks user where training data is if haven't already
            if self.__train_dir == "":
                self.__train_dir = self.get_inp("Where do I get training data? ")


            # Lists model architectures and asks which one they would like to train
            self.list_models()
    
            model_idx: int = self.get_inp(
                display="Select model type from above: ",
                expected=[i for i in range(len(self.__model_classes))]
                )

            model: nn.Module = self.__model_classes[model_idx]
            print(f"Model will be saved as: {model.__name__}\nIn folder: {self.__model_dir}")


            # Update Trainer class
            self.__trainer.update(
                name=model.__name__,
                model=model,
                save_folder=self.__model_dir
            )
            if self.__ttl == None: # Create traintest loader if not exists
                self.__ttl = TrainTestLoader(root_dir=self.__train_dir)


            # Train model
            self.train()

        # Prediction mode
        elif choice == "p":
            print("\nPrediction mode chosen")

            # Asks users where training data is if haven't already
            if self.__test_dir == "":
                self.__test_dir = self.get_inp("Where do I get testing data? ")


            # Lists model architectures and asks which one they would like to test
            self.list_models()

            model_idx: int = self.get_inp(
                display="Select model type from above: ",
                expected=[i for i in range(len(self.__model_classes))]
                )

            model: nn.Module = self.__model_classes[model_idx]


            # Lists saved models based off chosen model architecture and asks which one
            # they would like to test
            files = [f for f in os.listdir(self.__model_dir) if model.__name__ in f]
            
            # If there are no files found it will tell the user and not do any training.
            if len(files) != 0:
                for i in range(len(files)):
                    print(f"[{i}]: {files[i]}")


                saved_idx: int = self.get_inp(
                    display="Select saved model from above: ",
                    expected=[i for i in range(len(files))]
                )


                # Create instance of Trainer class
                self.__validator.update(
                    model=model,
                    path=f"{self.__model_dir}/{files[saved_idx]}"
                )

                if self.__fl == None: # Create full loader if not exists
                    self.__fl = FullLoader(root_dir=self.__test_dir)


                self.test() # Predict images using model
            else:
                print("Model weights not found")



    def train(self):
        """
        Trains model and outputs final accuracy and training/testing loss over epochs

        Keyword argument:
            model: Model architecture that will be trained
        """

        # Train model, tloss is train loss, and vloss is validation/test loss
        tloss, vloss = self.__trainer.train(self.__ttl)

        # Calculate difference between tloss and vloss over epochs
        loss_dif = [abs(t-vloss[i]) for i,t in enumerate(tloss)]

        # Finished training and creating graphs
        print("Training Finished!")

        # Creates an array of epoch numbers from 1 to n
        epochs = [e+1 for e in range(len(tloss))]

        # Plot training and validation loss loss difference between tloss and vloss
        grapher.line_chart(
            x={
                "Train loss": tloss,
                "Test loss": vloss,
                "Loss diff": loss_dif
            },
            y=epochs,
            xlabel="Epochs",
            ylabel="Loss",
            title="Loss over epochs"
        )



    def test(self):
        """
        Puts model into prediction mode and outputs accuracy over test dataset.

        Keyword arguments:
            model: Model architecture that will be used for prediction
            filename: Filename of the model weights that will be loaded into model architecture
        """

        print("Starting test...")

        # Run model through images in self.test_dir and gets results
        results = self.__validator.check_acc(self.__fl)

        # Save results to CSV
        self.__validator.save_to_csv(results, "results.csv")
        print("Results saved in results.csv")

        # Show results in graphical form
        self.show_test_results(results)



    def show_test_results(self, results: dict, is_wingdings=False):
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
        confidence_sum = [0]*len(CHARSET)
        correct = [0]*len(CHARSET)
        confidence_correct = []
        confidence_wrong = []
        totals = [0]*len(CHARSET)
        font_counts = {}


        # Iterates through each entry in results and notes down prediction, and the actual
        # character as well as incrementing correct and totals as well as adding to sum of
        # confidence per character.
        for i in range(len(results["filename"])):
            font = results["filename"][i].split("_")[0]
            pred = results["predicted letter"][i] # Precicted letter
            actual = results["actual letter"][i] # Actual letter
            result = results["result"][i] # Result of prediction "Pass" or "Fail"
            conf = float(results["confidence score"][i]) # Confidence score
            j = CHARSET.index(actual) # Index of actual letter within CHARSET

            if font not in font_counts:
                font_counts[font] = {"correct": 0, "total": 0}

            # Character/letter has to be turned into index here to match neural network output
            y_test.append(char_to_idx[actual]) # Appends actual letter to y_test
            y_pred.append(char_to_idx[pred]) # Appends predicted letter to y_pred

            # Adds confidence to sum at index j
            confidence_sum[j] += conf


            # If the result was a pass then it will increment correct at index j and font
            if result == "Pass":
                correct[j] += 1
                font_counts[font]["correct"] += 1
                confidence_correct.append(conf)
            else:
                confidence_wrong.append(conf)


            # Adds total at index j and font
            totals[j] += 1
            font_counts[font]["total"] += 1


        # Calculates confidence avg and percentage of correct predictions up to 2 d.p. and puts
        # them into a dictionary for plotting
        char_means = {
            "confidence_avg": [100*round(conf/totals[i], 2) for i,conf in enumerate(confidence_sum)],
            "correct_pct": [100*round(count/totals[i], 2) for i,count in enumerate(correct)]
        }


        # Calculate accuracy per class
        font_keys = font_counts.keys()
        font_acc = [
            100*round(
            font_counts[k]["correct"]/font_counts[k]["total"], 
            2
            ) for k in font_keys
        ]


        print(f"Avg confidence level for accurate results: {100 * sum(confidence_correct)/len(confidence_correct):.2f}%")
        print(f"Avg confidence level for inaccurate results: {100 * sum(confidence_wrong)/len(confidence_wrong):.2f}%")


        # Change label display if wingdings model
        if is_wingdings:
            display_labels = CHARSET_W
        else:
            display_labels = CHARSET


        # Show confusion matrix based of y_test and y_pred
        grapher.confusionmatrix_chart(
            y_test,
            y_pred,
            labels=display_labels
        )

        # Show confidence and accuracy for characters 
        grapher.groupedbar_chart(
            char_means,
            labels=display_labels,
            xlabel="Character",
            ylabel="%",
            title="Confidence and Accuracy per Character"
        )

        # Show accuracy for fonts
        grapher.bar_chart(
            x=font_acc, 
            y=font_keys, 
            xlabel="Font",
            ylabel="%",
            title="Accuracy per Font",
            rotation=85
        )



if __name__ == "__main__":
    main = Main(
        model_dir="release", 
        train_dir="character_images_wingdings",
        test_dir="dataset/final_test"
    )
    main.loop()