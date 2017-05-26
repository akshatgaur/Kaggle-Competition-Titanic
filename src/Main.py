import sys
from DataPreprocessing import preprocess
import pandas as pd

if __name__ == '__main__':

    # Perform preprocessing on the data set provided

    # get train data from the file path provided
    train_filepath = sys.argv[1]
    test_filepath = sys.argv[2]
    obj = preprocess()
    feature_names, train_data, test_data = obj.read_file(train_filepath, test_filepath)

    # perform feature modelling
    train_data = obj.feature_processing(train_data,train=True)
    test_data = obj.feature_processing(test_data)

    train_data.to_csv('train_processed.csv', encoding='utf-8', index=False, sep=";", header=None)
    test_data.to_csv('test_processed.csv', encoding='utf-8', index=False, sep=";", header=None)

    train_data = train_data.values.tolist()
    #print train_data

