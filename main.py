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

    def list_models(self):
        for i in range(len(self.model_classes)):
            print(f"[{i}]: {self.model_classes[i].__name__}")

    def loop(self):
        while True:
            user_inp = self.get_inp(
                display="\nWhat would you like to do?\n(T for train, P for predict, E for exit)\n",
                expected=["t", "p", "e"]
                )

            if user_inp == "e":
                break

            self.select_train_pred(user_inp)

    def get_inp(self, display: str, expected=[]):
        while True:
            user_inp = input(display).strip()

            try:
                user_inp = int(user_inp)
            except ValueError:
                user_inp = user_inp.lower()

            if expected == [] or user_inp in expected:
                return user_inp
            else:
                print("Invalid input")

    def select_train_pred(self, choice):
        if self.model_dir == "":
            self.model_dir = self.get_inp("Which folder do I read and write models from and to? ")

        if choice == "t":
            print("\nTraining mode chosen")

            if self.data_dir == "":
                self.data_dir = self.get_inp("Which folder contains training data? ")

            self.list_models()
            model_idx = self.get_inp(
                display="Select model type from above: ",
                expected=[i for i in range(len(self.model_classes))]
                )
            model = self.model_classes[model_idx]
            print(f"Model will be saved as: {model.__name__}\nIn folder: {self.model_dir}")

            self.train(model)
        elif choice == "p":
            print("\nPrediction mode chosen")

            if self.test_dir == "":
                self.test_dir = self.get_inp("Which folder contains images that needs to be predicted? ")

            self.list_models()
            model_idx = self.get_inp(
                display="Select model type from above: ",
                expected=[i for i in range(len(self.model_classes))]
                )
            model = self.model_classes[model_idx]
            
            files = [f for f in os.listdir(self.model_dir) if model.__name__ in f]
            for i in range(len(files)):
                print(f"[{i}]: {files[i]}")

            saved_idx = self.get_inp(
                display="Select saved model from above: ",
                expected=[i for i in range(len(files))]
            )

            self.test(model, files[saved_idx])

    def train(self, model):
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
        
    def test(self, model, filename):
        validator = Validator(
            path=f"{self.model_dir}/{filename}",
            model=model,
            root_dir=self.test_dir
        )

        print("Starting test...")
        results = validator.check_acc()
        validator.save_to_csv(results, "results.csv")
        print("Results saved in results.csv")

        self.show_test_results(results)

    def show_test_results(self, results):
        y_test = []
        y_pred = []
        confidence_sum = [0 for _ in CHARSET]
        correct = [0 for _ in CHARSET]
        totals = [0 for _ in CHARSET]

        for i in range(len(results["filename"])):
            # filename = results["filename"][i]
            pred = results["predicted letter"][i]
            actual = results["actual letter"][i]
            result = results["result"][i]
            conf = float(results["confidence score"][i])
            j = CHARSET.index(actual)

            y_test.append(char_to_idx[actual])
            y_pred.append(char_to_idx[pred])

            confidence_sum[j] += float(conf)

            if result == "Pass":
                correct[j] += 1

            totals[j] += 1

        char_means = {
            "confidence_avg": [round(conf/totals[i], 2) for i,conf in enumerate(confidence_sum)],
            "correct_pct": [round(count/totals[i], 2) for i,count in enumerate(correct)]
        }

        ConfusionMatrixDisplay.from_predictions(
            y_test, 
            y_pred,
            include_values=False,
            display_labels=CHARSET,
            normalize="true",
            cmap="Blues"
        )
        plt.show()

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