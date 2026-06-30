import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay #https://stackoverflow.com/questions/74020233/how-to-plot-confusion-matrix-in-pytorch



def line_chart(x: dict, y: list, xlabel="", ylabel="", title=""):
    for key, item in x.items():
        plt.plot(y, item, label=key)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def confusionmatrix_chart(y_test: list, y_pred: list, labels: list):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        include_values=False,
        display_labels=labels,
        normalize="true",
        cmap="Blues"
    )
    
    plt.show()


def groupedbar_chart(data: dict, labels: list, xlabel="", ylabel="", title="", rotation=0):
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    fig, ax = plt.subplots(layout="constrained")

    res = ax.grouped_bar(
        data,
        tick_labels=labels,
        group_spacing=1
    )

    for container in res.bar_containers:
        ax.bar_label(container, padding=3)

    plt.tick_params("x", rotation=rotation)
    plt.xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.legend(loc="outside left upper")
    plt.show()


def bar_chart(x: list, y: list, xlabel="", ylabel="", title="", rotation=0):
    plt.bar(y, x)

    plt.tick_params("x", rotation=rotation)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()