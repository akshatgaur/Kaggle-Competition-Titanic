import random
import pandas as pd
import numpy as np
import DataPreprocessing

# Split input file into train and test
def main(file_name):
    random.seed(27)

    cols = pd.read_csv(file_name, nrows=1, sep=";").columns
    cols_list = cols.tolist()

    df = pd.read_csv(file_name, sep=';')
    mask = np.random.rand(len(df)) <= 0.80

    train = df[mask]
    test = df[~mask]

    print len(test)
    print len(train)

    train.to_csv('train.csv', encoding='utf-8', index=False, sep=";", header=None)
    test.to_csv('test.csv', encoding='utf-8', index=False, sep=";", header=None)

    # del test['class']
    test.to_csv('test_copy.csv', encoding='utf-8', index=False, sep=";", header=None)

    return 'train.csv', 'test.csv', 'test_copy.csv'


# Split the data into N files which will be input for each mapper
def split_train(train, N, test_file, prefix):
    random.seed(27)
    path = 'trainFile_'
    input_file = []
    cols = pd.read_csv(train, nrows=1, sep=";").columns
    cols_list = cols.tolist()
    df = pd.read_csv(train, sep=';')

    for i in range(N):
        path_i = path + str(i) + '.csv'
        mask = np.random.rand(len(df)) <= 0.67
        train_i = df[mask]
        train_i.to_csv(path_i, encoding='utf-8', index=False, sep=";", header=None)
        input_file.append(prefix + path_i + " " + prefix + test_file)

    f = open('./mapper_input_file.txt', 'w')
    f.write("\n".join(input_file))
    return './mapper_input_file.txt'

if __name__ == '__main__':

    import sys

    # Perform preprocessing on the data set provided
    # get train data from the file path provided
    train_filepath = sys.argv[1]
    test_filepath = sys.argv[2]
    N = int(sys.argv[3])
    prefix = sys.argv[4]
    obj = DataPreprocessing.preprocess()
    feature_names, train_data, test_data = obj.read_file(train_filepath, test_filepath)

    # perform feature modelling
    train_data = obj.feature_processing(train_data, train=True)
    test_data = obj.feature_processing(test_data)

    train_data.to_csv('train_processed.csv', encoding='utf-8', index=False, sep=";", header=None)
    test_data.to_csv('test_processed.csv', encoding='utf-8', index=False, sep=";", header=None)

    # read train file
    train_file_name = 'train_processed.csv'
    test_file_name = 'test_processed.csv'
    file_object = open(train_file_name)
    try:
        file_context = file_object.read()
    finally:
        file_object.close()

    file_context = file_context.split('\n')
    train_data = []
    for line in file_context:
        st = line.split(';')
        train_data.append(st)

    # split data into multiple files for mappers
    mapper_file_name = split_train(train_file_name, N, test_file_name, prefix)


