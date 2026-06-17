class Main:
    def __init__(self):
        pass

    def loop(self):
        while True:
            user_inp = str(input("\nTraining or prediction mode?\n(T for train, P for predict)\n")).lower()
            self.select_train_pred(user_inp)

    def select_train_pred(self, choice):
        if choice == "t":
            print("Training mode chosen")
            self.train()
        elif choice == "p":
            print("Prediction mode chosen")
        else:
            print("Invalid input")

    def train(root_dir):
        root_dir



if __name__ == "__main__":
    main = Main()
    main.loop()