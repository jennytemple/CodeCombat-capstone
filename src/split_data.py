import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_files(path):
    event_header = ['Created', 'User Id', 'Event Name', 'Level',
                    'raw_event1', 'raw_event2', 'raw_event3', 'raw_event4']

    df_users = pd.read_csv(path + 'users.csv')
    if 'Unnamed: 0' in df_users.columns:
        df_users.drop('Unnamed: 0', axis=1, inplace=True)
    df_levels = pd.read_csv(path + 'levels.csv')
    df_events = pd.read_csv(path + 'events.csv', skiprows=1,
                            names=event_header, error_bad_lines=False, warn_bad_lines=True)
    return (df_users, df_levels, df_events)


if __name__ == '__main__':
    march_path = '../../data/march/'
    path = march_path
    df = pd.read_csv(path + 'users.csv')
    df_train, df_test = train_test_split(df)

    df_train.to_csv(path + 'train/users.csv')
    df_test.to_csv(path + 'train/dusers.csv')
