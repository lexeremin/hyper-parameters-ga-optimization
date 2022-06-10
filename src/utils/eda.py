import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def class_disribution(data):
    datas=pd.DataFrame(
        {
            "Y_train": data["Y_train"].astype(int).reshape((data["Y_train"].shape[0])),
            "Y_test": data["Y_test"].astype(int).reshape((data["Y_test"].shape[0]))
        }, 
        columns=["Y_train", "Y_test"]
    )
    # datas=pd.DataFrame(
    #     {
    #         "Y_train": data["Y_train"].astype(int).reshape((data["Y_train"].shape[0])),
    #     }
    # )
    # datas.plot.bar(stacked=True)
    print(datas.head())
    # sns.histplot(x=0, data=datas)

    # plt.figure()
    # plt.hist([ 
    #         data["Y_train"].astype(int),
    #         data["Y_test"].astype(int)
    #         ], stacked=True, density=True)
    plt.show()

def main():

    ...

if __name__ == "__main__":
    main()
