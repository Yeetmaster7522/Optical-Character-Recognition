import models
from training import Trainer
from validation import Validator

class Main:
    def __init__(self, model_dir="", data_dir="", predict_dir=""):
        print("WELCOME")
        
        self.model_classes = [
            models.ocr_v1,
            models.ocr_v2,
            models.ocr_v3,
            models.ocr_v4,
        ]
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.predict_dir = predict_dir

        self.get_directories()

    def get_directories(self):
        if self.model_dir == "":
            self.model_dir = str(input("Which folder do I read and write models from and to? "))
        
        if self.data_dir == "":
            self.data_dir = str(input("Which folder contains training data? "))

        if self.predict_dir == "":
            self.predict_dir = str(input("Which folder contains images that need to be predicted? "))

    def list_models(self):
        for i in range(len(self.model_classes)):
            print(f"[{i}]: {self.model_classes[i].__name__}")

    def loop(self):
        while True:
            user_inp = str(input("\nTraining or prediction mode?\n(T for train, P for predict)\n")).lower()

            if user_inp == "exit":
                break

            self.select_train_pred(user_inp)

    def select_train_pred(self, choice):
        if choice == "t":
            print("Training mode chosen")
            self.train()
        elif choice == "p":
            print("Prediction mode chosen")
            self.predict()
        else:
            print("Invalid input")

    def train(self):
        self.list_models()
        model_i = int(input("Select model type from above: "))
        model = self.model_classes[model_i]

        name = str(input("What name would you like to give the model? "))

        trainer = Trainer(
            name=name,
            model=model,
            save_folder=self.model_dir,
            root_dir=self.data_dir
        )
        losses = trainer.train()
        print("Training Finished!")
        trainer.plot(losses)

    def predict(self):
        self.list_models()
        model_i = int(input("Select model type from above: "))
        model = self.model_classes[model_i]()
        # NOTE NEED TO LINK TO VALIDATION.PY



if __name__ == "__main__":
    main = Main(
        model_dir="models/models_F", 
        data_dir="dataset/character_images_no_noise"
    )
    main.loop()