import pickle as pkl
import numpy as np
import os


def main():
    FOLDER = "data/tl_train/U6T7/xu2015TrainHl60/set0/10_fold/0_fold/"
    FOLDER_2 = "data/tl_train/U6T7/xu2015TrainHl60/set1/10_fold/0_fold/"
    FILE_NAME = "X_biofeat_train.pkl"

    with open(os.path.join(FOLDER, FILE_NAME), "rb") as fp:
        X_biofeat_train = pkl.load(fp)

    with open(os.path.join(FOLDER_2, FILE_NAME), "rb") as fp:
        X_biofeat_train_2 = pkl.load(fp)

    print(X_biofeat_train.shape)
    print(X_biofeat_train_2.shape)

    # Save first 100 rows of each numpy array into a csv file
    np.savetxt("X_biofeat_train.csv", X_biofeat_train[:100, :], delimiter=",")
    np.savetxt("X_biofeat_train_2.csv", X_biofeat_train_2[:100, :], delimiter=",")


if __name__ == "__main__":
    main()