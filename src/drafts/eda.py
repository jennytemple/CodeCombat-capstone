import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def filter_missing(df):
    return df.dropna(axis=0, how='any')

def categorize(level_nums, level_names=None, num_cats=4):

    '''
    For now, use number of levels, but later do by last campaign
    campaigns = pd.read_csv('../data/campaign_list.csv')
    '''
    if num_cats == 2:
        bins = np.array([5, 20, 999])
    if num_cats == 4:
        bins = np.array([5, 12, 39, 100, 999])

    y = np.digitize(level_nums, bins)
    return y


def drop_fields(df):
    captured_in_target = ['Playtime (s)', 'last_event_date', 'last_action', 'last_level_name', 'active_time_days', 'data_through', 'activity_gap_days', 'Level', 'last_level_time_s', 'daygap', 'was_completed']
    not_useful = ['Unnamed: 0','Id', 'Date Joined']
    too_sparse = ['Gender', 'How old are you?','Want to be a programmer?', 'How long have you been programming?', 'How hard is CodeCombat?','How did you hear about CodeCombat?','Gender?', 'Favorite programming language?', 'Early bird or night owl?', 'What polls do you like?', 'Friends who code?', 'How fast is your internet?', 'After playing CodeCombat',' how interested are you in programming?', 'How likely are you to recommend CodeCombat?', "How likely that you'd recommend CodeCombat?"]
    df.drop(captured_in_target, axis=1, inplace=True)
    df.drop(not_useful, axis=1, inplace=True)
    df.drop(too_sparse, axis=1, inplace=True)
    return df

def dummify(X):
    countries = pd.get_dummies(X['Country'])
    X[countries.columns] = countries

    languages = pd.get_dummies(X['Code Language'])
    X[languages.columns] = languages

    X.drop(['Country', 'Code Language'], axis=1, inplace=True)
    return X

if __name__=='__main__':
    sample_path = '../data/sample/'
    august_path = '../data/august/'
    march_path = '../data/march/'

    path = march_path
    df = pd.read_csv(path+'post_processed_users.csv')

    df = drop_fields(df)
    df = filter_missing(df)

    y = df.pop('Levels Completed').values
    #y2 = df.pop('last_level_name').values
    y = categorize(y, 4)


    X = dummify(df).values

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’
    multi_lr = LogisticRegression(multi_class='multinomial',solver='sag')
    multi_lr.fit(X_train, y_train)
    multi_lr.score(X_train, y_train)

    df.isnull().any()
    df['avg_num_days_per_level'].isnull().value_counts()
    df['avg_L6_playtime_s'].isnull().value_counts()
