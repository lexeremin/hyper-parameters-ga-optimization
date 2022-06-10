import numpy as np
import pandas as pd
# import json
# from pathlib import Path
# import os

_TEST_DICT = {'TRAIN_FILES': ['./data/ECG200/ECG200_TRAIN.tsv', './data/ECG5000/ECG5000_TRAIN.tsv', './data/Strawberry/Strawberry_TRAIN.tsv', './data/Fish/Fish_TRAIN.tsv', './data/Haptics/Haptics_TRAIN.tsv', './data/Plane/Plane_TRAIN.tsv', './data/FacesUCR/FacesUCR_TRAIN.tsv'], 'TEST_FILES': ['./data/ECG200/ECG200_TEST.tsv', './data/ECG5000/ECG5000_TEST.tsv', './data/Strawberry/Strawberry_TEST.tsv', './data/Fish/Fish_TEST.tsv', './data/Haptics/Haptics_TEST.tsv', './data/Plane/Plane_TEST.tsv', './data/FacesUCR/FacesUCR_TEST.tsv']}
# def get_project_root() -> Path:
#     return os.path.join(Path(__file__).parent.parent, "configs/")

# def get_ts_data():
#     config = 'data-config.json'
#     try:
#         with open(os.path.join(get_project_root(), config)) as fd:
#             data = json.load(fd)
#     except IOError:
#         print("Couldn't load data-config.json")
#         exit(0)
#     return data

def data_loader(i = 0, data = {}):
    if data:
        if i < len(data['TRAIN_FILES']):
            #TRAIN
            df = pd.read_csv(data['TRAIN_FILES'][i], header=None, sep='\t')
            df.dropna(axis=1, how='all', inplace=True)
            # fill all missing columns with 0
            df.fillna(0, inplace=True)
            df[df.columns] = df[df.columns].astype(np.float32)
            Y_train = df[[0]].values
            nb_classes = len(np.unique(Y_train))
            Y_train = np.around((Y_train - Y_train.min()) / (Y_train.max() - Y_train.min()) * (nb_classes - 1))
            # drop labels column from train set X
            df.drop(df.columns[0], axis=1, inplace=True)
            X_train = df.values
            #TEST
            df = pd.read_csv(data['TEST_FILES'][i], header=None, sep='\t')
            df.dropna(axis=1, how='all', inplace=True)
            # fill all missing columns with 0
            df.fillna(0, inplace=True)
            df[df.columns] = df[df.columns].astype(np.float32)
            Y_test = df[[0]].values
            nb_classes = len(np.unique(Y_test))
            Y_test = np.around((Y_test - Y_test.min()) / (Y_test.max() - Y_test.min()) * (nb_classes - 1))
            # drop labels column from train set X
            df.drop(df.columns[0], axis=1, inplace=True)
            X_test = df.values
            # print(type(X_train))
            # print(Y_test)
            return {"X_train": X_train, "Y_train": Y_train, "X_test": X_test, "Y_test": Y_test}
            # return X_train
        else:
            print(f"There is no data at index {i}")
            exit(0)
    else:
        print("Data fetching error")
        exit(0)

def main():
    print(data_loader(4, _TEST_DICT))

if __name__ == '__main__':
    main()