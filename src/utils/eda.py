import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def class_distribution(data, classname):
    _, train_counts = np.unique(np.around(data["Y_train"]), return_counts=True)
    _, test_counts = np.unique(np.around(data["Y_test"]), return_counts=True)
    print(train_counts, test_counts)

    processed_data=pd.DataFrame(
        {
            "Обучающая выборка": train_counts,
            "Тестовая выборка": test_counts
        }, 
        columns=["Обучающая выборка", "Тестовая выборка"]
    )
    print(processed_data.head())
    processed_data.plot.bar(rot=0, xlabel="Номер класса", ylabel="Размер выборки", title=classname)
    plt.show()

def train_test_distribution(data, classname):

    processed_data=pd.DataFrame(
        {
            "Y_train": data["Y_train"].shape[0],
            "Y_test": data["Y_test"].shape[0]
        }, 
        index=[""],
        columns=["Y_train", "Y_test"]
    )
    print(processed_data.head())
    processed_data.plot.bar(rot=0, label=classname)
    plt.show()

def main():

    ...

if __name__ == "__main__":
    main()
