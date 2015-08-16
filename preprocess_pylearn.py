import pylearn2

from pylearn2.datasets.csv_dataset import CSVDataset
from pylearn2.datasets import preprocessing
from pylearn2.utils import serial

if __name__ == "__main__":
    train = CSVDataset("train_bin.csv")
    test = CSVDataset("test_bin.csv")

    pipeline = preprocessing.Pipeline()
    pipeline.items.append(preprocessing.GlobalContrastNormalization(\
        sqrt_bias=10.0, use_std=True))
    pipeline.items.append(preprocessing.ZCA())

    train.apply_preprocessor(preprocessor=pipeline, can_fit=True)
    test.apply_preprocessor(preprocessor=pipeline, can_fit=True)

    train.use_design_loc("train_preprocessed_design.npy")
    test.use_design_loc("test_preprocessed_design.npy")
    serial.save("train_preprocessed.pkl", train)
    serial.save("test_preprocessed.pkl", test)
