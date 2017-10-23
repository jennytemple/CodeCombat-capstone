import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def turn_off_warnings():
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn


def filter_missing(df):
    '''
    removes rows of data with any missing values
    note: missing values may be from the small number of users who deviate from the typical level progression. Data owners stated that ignoring these users may be best
    '''
    return df.dropna(axis=0, how='any')


def categorize_by_level_num(level_nums, num_cats):
    '''
    For now, use number of levels, but later do by last campaign
    campaigns = pd.read_csv('../data/campaign_list.csv')
    general order of campaings: dungeon, campaign-web-dev-1, campaign-game-dev-1, forest, campaign-web-dev-2, campaign-game-dev-2, desert, mountain, glacier
    '''
    name = "categories by level"
    if num_cats == 2:
        # 1st bin: x s.t. value0 <= x < value1
        bins = np.array([0, 12, 999])
        print bins
    if num_cats == 3:
        # levels 20 to 21 is a pretty big drop off
        bins = np.array([0, 12, 21, 999])
    if num_cats == 4:
        # levels 20 to 21 is a pretty big drop off
        bins = np.array([0, 12, 30, 100, 999])

    y = np.digitize(level_nums, bins)
    return y, name


def categorize_by_campaign(y, num_cats):
    '''
    Use to predict where in game the player will end instead of number of levels played
    campaigns = pd.read_csv('../data/campaign_list.csv')
    general order of campaigns: dungeon, campaign-web-dev-1, campaign-game-dev-1, forest, campaign-web-dev-2, campaign-game-dev-2, desert, mountain, glacier, other(can be anywhere)
    '''
    name = "categories by campaign"
    if num_cats == 2:
        early = ['dungeon']
        y = y.apply(lambda row: "early_churn" if row in [
                    'dungeon'] else "later_churn")

    if num_cats == 3:
        early = ['dungeon']
        mid = ['campaign-web-dev-1', 'campaign-game-dev-1', 'forest']
        y = y.apply(lambda row: "early_churn" if row in early else (
            "mid_churn" if row in mid else "later_churn"))

    if num_cats == 4:
        early = ['dungeon']
        mid = ['campaign-web-dev-1', 'campaign-game-dev-1', 'forest']
        late = ['campaign-web-dev-2', 'campaign-game-dev-2',
                'desert', 'mountain', 'glacier']
        y = y.apply(lambda row: "early_churn" if row in early else (
            "mid_churn" if row in mid else ("later_churn" if row in late else "other")))
    # if num_cats == 4:
    #     early = ['dungeon']
    #     mid = ['forest']
    #     late = ['campaign-web-dev-1', 'campaign-game-dev-1','campaign-web-dev-2', 'campaign-game-dev-2', 'desert', 'mountain', 'glacier']
    #     y = y.apply(lambda row: "early_churn" if row in early else ("mid_churn" if row in mid else ("later_churn" if row in late else "other")))
    # could also investigate campaign vs dev
    return y.values, name


def drop_unmodeled_fields(df):

    dates = ['date_completed_first_six', 'Date Joined']
    target_leakage = ['num_levels_completed_in_first_six']
    eda_only_fields = ['last_event_date',
                       'avg_num_days_per_level',
                       'last_action',
                       'active_time_days',
                       'total_play_time',
                       'last_level_was_practice',
                       'last_level_name',
                       'activity_gap_days',
                       'data_through',
                       'last_level_was_completed',
                       'daygap',
                       'last_level_played',
                       'avg_play_time_per_level_s',
                       'last_level_time_s']

    too_sparse = ['How likely are you to recommend CodeCombat?',
                  'How hard is CodeCombat?',
                  'What polls do you like?',
                  'How interested are you in programming?',
                  'How did you hear about CodeCombat?',
                  'Early bird or night owl?',
                  'How fast is your internet?',
                  'Friends who code?',
                  "How likely that you'd recommend CodeCombat?",
                  'Want to be a programmer?',
                  'Gender',
                  'Favorite programming language?',
                  'How long have you been programming?',
                  'Gender?']

    not_useful = ['Id', 'Unnamed: 0']

    df.drop(dates, axis=1, inplace=True)
    df.drop(target_leakage, axis=1, inplace=True)
    df.drop(eda_only_fields, axis=1, inplace=True)
    df.drop(too_sparse, axis=1, inplace=True)
    df.drop(not_useful, axis=1, inplace=True)

    return df


def fix_target_and_drop_target_fields(df, target):
    '''
    return identified target field and drop other target-related fields
    '''
    target_fields = ['last_campaign_started',
                     'last_level_started', 'Levels Completed']

    y = df.pop(target)
    for level in target_fields:
        if level != target:
            df.drop(level, axis=1, inplace=True)

    return df, y


def dummify_with_countries(X):
    countries = pd.get_dummies(X['Country'])
    X[countries.columns] = countries

    ages = pd.get_dummies(X['How old are you?'])
    X[ages.columns] = ages

    X.drop(['Country', 'How old are you?'], axis=1, inplace=True)
    return X


def dummify_no_countries(X):
    ages = pd.get_dummies(X['How old are you?'])
    X[ages.columns] = ages

    X.drop(['Country', 'How old are you?'], axis=1, inplace=True)
    return X


def extreme_filter(X):
    '''
    Make a very simple version of X to test that model is running
    '''
    X = X[['avg_first_six_playtime_s']]
    return X


def random_feature(X):
    X = np.random.rand(X.shape[0], 1)
    return X


def print_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)

    f1_score_micro = f1_score(y_true, y_pred, average='micro')
    f1_score_none = f1_score(y_true, y_pred, average=None)
    f1_score_macro = f1_score(y_true, y_pred, average='macro')
    f1_score_weighted = f1_score(y_true, y_pred, average='weighted')

    print "\taccuracy = {}, \n\tprecision = {}, \n\trecall = {}".format(accuracy, precision, recall)
    print "\tF1 score for each class = {} \n\tF1 score, micro = {} \n\tF1 score, macro = {} \n\tF1 score, weighted {}".format(f1_score_none, f1_score_micro, f1_score_macro, f1_score_weighted)


def model_multi_log_reg(X_train_scaled, y_train, name, X_test, y_test):
    '''
    Model with a multinomial logistic regression and print results
    '''
    # For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’
    multi_lr = LogisticRegression(
        multi_class='multinomial', solver='sag', max_iter=1000, class_weight='balanced')
    multi_lr.fit(X_train_scaled, y_train)

    print "On TRAIN data:"
    y_predict = multi_lr.predict(X_train_scaled)
    print "\nThe Multinomial logistic regression modeled {} with {} classes yielded a model with:".format(name, num_labels)
    print_scores(y_train, y_predict)

    print "On TEST data:"
    y_predict = multi_lr.predict(X_test)
    print "\nThe Multinomial logistic regression modeled {} with {} classes yielded a model with:".format(name, num_labels)
    print_scores(y_test, y_predict)

    return multi_lr.coef_


def model_random_forest(X_train, y_train, name, X_test, y_test):
    '''
    Model with a random forest and print results
    '''
    rand_forest = RandomForestClassifier(class_weight="balanced")
    rand_forest.fit(X_train, y_train)

    print "On TRAIN data:"
    y_predict = rand_forest.predict(X_train)
    print "\nThe Random Forest model {} with {} classes yielded a model with:".format(name, num_labels)
    print_scores(y_train, y_predict)

    print "On TEST data:"
    y_predict = rand_forest.predict(X_test)
    print "\nThe Random Forest model {} with {} classes yielded a model with:".format(name, num_labels)
    print_scores(y_test, y_predict)

    return rand_forest.feature_importances_


def rank_features(cols, coefs):
    cols = np.array(cols)
    cols = cols.reshape(cols.shape[0], 1)
    coefs = np.around(coefs, decimals=4)
    coefs = coefs.reshape(cols.shape[0], 1)
    coefs = coefs.reshape(coefs.shape[0], 1)
    abs_coefs = np.abs(coefs)

    results = np.concatenate((coefs, cols, abs_coefs), axis=1)
    return results[np.argsort(results[:, 2])]


def compare_with_random_mlr_model(orig_X, y_train):
    X = random_feature(orig_X)
    rando_pred = model_multi_log_reg(X, y_train, "random feature")


if __name__ == '__main__':

    turn_off_warnings()

    sample_path = '../../data/sample/'
    august_path = '../../data/august/'
    march_path = '../../data/march/'

    path = march_path
    df = pd.read_csv(path + 'post_processed_users.csv')

    df = drop_unmodeled_fields(df)
    df = filter_missing(df)

    target = 'Levels Completed'
    #target = 'last_campaign_started'
    num_labels = 2
    df, y = fix_target_and_drop_target_fields(df, target)
    print list(df.columns)
    y, name = categorize_by_level_num(y, num_labels)
    #y, name = categorize_by_campaign(y, num_labels)

    df = dummify_with_countries(df)
    #df = dummify_no_countries(df)
    features = list(df.columns)

    X = df.values
    # X = extreme_filter(X).values

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    ''' Multinomial Logistic Regression '''
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    mlr_feature_importance = model_multi_log_reg(
        X_train_scaled, y_train, name, X_test, y_test)
    mlr_feature_importance_readable = rank_features(
        features, mlr_feature_importance)[::-1]
    print mlr_feature_importance_readable

    ''' Random Forest'''
    rf_feature_importance = model_random_forest(
        X_train, y_train, name, X_test, y_test)

    rf_feature_importance_readable = rank_features(
        features, rf_feature_importance)[::-1]
    print rf_feature_importance_readable

    '''Random Model'''
    # compare_with_random_mlr_model(X_train_scaled, y_train)
    print "\nRandom Model"
    random_X_train = random_feature(X_train)
    random_X_test = random_feature(X_test)
    random_y_pred = model_random_forest(
        random_X_train, y_train, name, random_X_test, y_test)

    '''*** Results *** '''
    '''
    using campaigns, 4

    '''
    '''
    using num levels, 4

    '''
    '''
    using campaigns, 2
    '''
    '''
    using levels, 2
    '''
    '''
    using 3 campaigns
    '''
