import os

import models
from training import Trainer
from validation import Validator

class Main:
    def __init__(self, model_dir="", data_dir="", test_dir=""):
        print("WELCOME")
        
        self.model_classes = [
            models.ocr_v1,
            models.ocr_v2,
            models.ocr_v3,
            models.ocr_v4,
        ]
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.test_dir = test_dir

        self.get_directories()

    def get_directories(self):
        if self.model_dir == "":
            self.model_dir = str(input("Which folder do I read and write models from and to? "))
        
        if self.data_dir == "":
            self.data_dir = str(input("Which folder contains training data? "))

        if self.test_dir == "":
            self.test_dir = str(input("Which folder contains images that need to be predicted?\nLeave blank if using same folder for training data "))
            if self.test_dir == "":
                self.test_dir == self.data_dir

    def list_models(self):
        for i in range(len(self.model_classes)):
            print(f"[{i}]: {self.model_classes[i].__name__}")

    def loop(self):
        while True:
            user_inp = str(input("\nWhat would you like to do?\n(T for train, P for predict, E for exit)\n")).lower()

            if user_inp == "e":
                break

            self.select_train_pred(user_inp)

    def select_train_pred(self, choice):
        if choice == "t":
            print("Training mode chosen")
            self.train()
        elif choice == "p":
            print("Prediction mode chosen")
            self.test()
        else:
            print("Invalid input")

    def train(self):
        self.list_models()
        model_idx = int(input("Select model type from above: "))
        model = self.model_classes[model_idx]

        print(f"Model will be saved as: {model.__name__}\nIn folder: {self.model_dir}")

        trainer = Trainer(
            name=model.__name__,
            model=model,
            save_folder=self.model_dir,
            root_dir=self.data_dir
        )
        losses = trainer.train()
        print("Training Finished!")
        trainer.plot(losses)

    def test(self):
        self.list_models()
        model_idx = int(input("Select model type from above: "))
        model = self.model_classes[model_idx]
        
        files = [f for f in os.listdir(self.model_dir) if model.__name__ in f]
        for i in range(len(files)):
            print(f"[{i}]: {files[i]}")

        saved_idx = int(input("Select saved model from above: "))

        validator = Validator(
            path=f"models/models_F/{files[saved_idx]}",
            model=model,
            root_dir=self.test_dir
        )



if __name__ == "__main__":
    main = Main(
        model_dir="models/models_F", 
        data_dir="dataset/character_images_no_noise"
    )
    main.loop()