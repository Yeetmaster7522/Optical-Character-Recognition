import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #https://stackoverflow.com/questions/74020233/how-to-plot-confusion-matrix-in-pytorch

import models
from training import Trainer
from validation import Validator
from dataloader import TrainTestLoader as TTL
from dataloader import CHARSET, char_to_idx

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
            test_dir = str(input("Which folder contains images that need to be predicted?\nLeave blank if using same folder for training data "))

            if test_dir == "":
                self.test_dir = self.data_dir
            elif test_dir != "":
                self.test_dir = test_dir

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
        tloss, vloss = trainer.train()
        print("Training Finished!")

        epochs = [e+1 for e in range(len(tloss))]

        plt.plot(epochs, tloss, color="red", label="Train loss")
        plt.plot(epochs, vloss, color="green", label="Test loss")
        
        plt.plot(epochs, [abs(t-vloss[i]) for i,t in enumerate(tloss)], color="blue", linestyle="-.", label="Loss diff")
        
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        
    def test(self):
        self.list_models()
        model_idx = int(input("Select model type from above: "))
        model = self.model_classes[model_idx]
        
        files = [f for f in os.listdir(self.model_dir) if model.__name__ in f]
        for i in range(len(files)):
            print(f"[{i}]: {files[i]}")

        saved_idx = int(input("Select saved model from above: "))

        validator = Validator(
            path=f"{self.model_dir}/{files[saved_idx]}",
            model=model,
            root_dir=self.test_dir
        )

        print("Starting test...")
        results = validator.check_acc()
        validator.save_to_csv(results, "results.csv")
        print("Results saved in results.csv")

        self.show_test_results(results)

    def show_test_results(self, results):
        filenames = results["filename"]
        y_test = [char_to_idx[chr(int(filename.removesuffix(".png").split("_")[2]))] for filename in filenames]
        y_pred = [char_to_idx[l] for l in results["predicted letter"]]

        ConfusionMatrixDisplay.from_predictions(
            y_test, 
            y_pred,
            include_values=False,
            display_labels=CHARSET,
            cmap="Blues"
        )
        plt.show()

        confidence_sum = [0 for _ in CHARSET]
        correct = [0 for _ in CHARSET]
        totals = [0 for _ in CHARSET]

        for i, c in enumerate(results["actual letter"]):
            char_idx = CHARSET.index(c)

            confidence_sum[char_idx] += float(results["confidence score"][i])
            if results["result"][i] == "Pass":
                correct[char_idx] += 1

            totals[char_idx] += 1

        char_means = {
            "confidence_avg": [round(conf/totals[i], 2) for i,conf in enumerate(confidence_sum)],
            "correct_pct": [round(count/totals[i], 2) for i,count in enumerate(correct)]
        }

        # print(char_means, correct, wrong)

        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
        fig, ax = plt.subplots(layout="constrained")

        res = ax.grouped_bar(char_means, tick_labels=CHARSET, group_spacing=1)
        for container in res.bar_containers:
            ax.bar_label(container, padding=3)

        ax.set_ylabel("%")
        ax.set_title("Confidence and Accuracy by Character")
        ax.legend(loc="upper left", ncols=2)
        # ax.set_ylim(0,100)
        plt.show()



if __name__ == "__main__":
    main = Main(
        model_dir="models/models_F", 
        data_dir="dataset/character_images_no_noise",
        test_dir="dataset/character_images_no_noise"
    )
    main.loop()