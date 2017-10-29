import pandas as pd
import numpy as np

if __name__ == '__main__':
    march_path = '../../data/march/'
    path = march_path
    df = pd.read_csv(path + 'users.csv')

    df_train = pd.read_csv(path + 'train/users.csv')
    df_test = pd.read_csv(path + 'test/users.csv')
