import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import pickle


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


def categorize_by_level_num(level_nums, num_cats, threshold=12):
    '''
    For now, use number of levels, but later do by last campaign
    campaigns = pd.read_csv('../data/campaign_list.csv')
    general order of campaings: dungeon, campaign-web-dev-1, campaign-game-dev-1, forest, campaign-web-dev-2, campaign-game-dev-2, desert, mountain, glacier
    '''
    name = "categories by level"
    if num_cats == 2:
        # 1st bin: x s.t. value0 <= x < value1
        # first most common 12 levels are in dungeon campaign
        # bins = np.array([0, 13, 999])
        bins = np.array([threshold, 999])
        print bins
    if num_cats == 3:
        # levels 20 to 21 is a pretty big drop off
        bins = np.array([0, 13, 21, 999])
    if num_cats == 4:
        # levels 20 to 21 is a pretty big drop off
        bins = np.array([0, 13, 30, 100, 999])

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

    return y.values, name


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


def drop_unnamed(df):
    if 'Unnamed: 0' in df.columns:
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
    return df


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
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # f1_score_none = f1_score(y_true, y_pred, average=None)
    # f1_score_macro = f1_score(y_true, y_pred, average='macro')
    # f1_score_weighted = f1_score(y_true, y_pred, average='weighted')

    print "\taccuracy = {}, \n\tprecision = {}, \n\trecall = {}\n\tf1-score = {}".format(accuracy, precision, recall, f1)
    # print "\tF1 score for each class = {} \n\tF1 score, micro = {} \n\tF1 score, macro = {} \n\tF1 score, weighted {}".format(f1_score_none, f1_score_micro, f1_score_macro, f1_score_weighted)
    print "\nThe confusion matrix is:"
    print "\tTN \tFP\n\tFN\tTP"
    print confusion_matrix(y_true, y_pred)


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
    #n_estimators, max_features="m", max_depth
    # tuned_parameters = [{'n_estimators': [10, 50, 100, 500], 'max_features': [
    #     "sqrt", "log2", 0.3, 0.5, 0.7], 'min_samples_leaf':[1, 3, 5]}]
    tuned_parameters = [{'n_estimators': [50], 'max_features': [
        "sqrt"], 'min_samples_leaf':[1]}]
    # tuned_parameters = [{'n_estimators': [10, 20, 30, 50, 75, 100]}]
    # rand_forest = RandomForestClassifier(class_weight="balanced")
    rand_forest = GridSearchCV(estimator=RandomForestClassifier(
        class_weight="balanced"), param_grid=tuned_parameters, scoring="accuracy")
    # rand_forest = GridSearchCV(estimator=RandomForestClassifier(
    #     class_weight="balanced"), param_grid=tuned_parameters, scoring="accuracy", verbose=100)
    rand_forest.fit(X_train, y_train)
    print "\nGrid search score: {}".format(rand_forest.best_score_)
    print "\nThe best parameters:\n\tn_estimators = {}\n\tmax_features = {}\n\tmin_samples_leaf = {}"\
        .format(rand_forest.best_estimator_.n_estimators, rand_forest.best_estimator_.max_features, rand_forest.best_estimator_.min_samples_leaf)

    tuned_rf = RandomForestClassifier(n_estimators=rand_forest.best_estimator_.n_estimators,
                                      max_features=rand_forest.best_estimator_.max_features,
                                      min_samples_leaf=rand_forest.best_estimator_.min_samples_leaf,
                                      class_weight="balanced")
    tuned_rf.fit(X_train, y_train)
    print "\nOn TRAIN data:"
    y_predict = tuned_rf.predict(X_train)
    print "The Random Forest model {} with {} classes yielded a model with:".format(name, num_labels)
    print_scores(y_train, y_predict)

    print "\nOn TEST data:"
    y_predict = tuned_rf.predict(X_test)
    print "The Random Forest model {} with {} classes yielded a model with:".format(name, num_labels)
    print_scores(y_test, y_predict)
    proba = tuned_rf.predict_proba(X_test)

    #fpr, tpr, thresholds = roc_curve(y_test, proba[:, 1])
    auc = roc_auc_score(y_test, proba[:, 1])
    print "\nThe area under the roc curve is: {}\n".format(auc)

    pickle.dump(tuned_rf, open("test.p", "wb"))

    return tuned_rf.feature_importances_, proba


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
    march_path = '~/galvanize/project/data/march/train/'
    tiny_sample_path = '../../data/tiny_sample/'

    # possibilities for level_predict: [10,15,30,60,100]
    level_predict = 13
    csv_name = 'model_predict_at_' + str(level_predict) + '.csv'

    path = march_path
    df = pd.read_csv(path + csv_name)

    df = drop_unnamed(df)  # unnamed field when read in csv
    df = filter_missing(df)

    # temp!
    # df.drop(['luachunk_20', 'pythonchunk_20',
    #  'javascriptchunk_20'], axis=1, inplace=True)

    target = 'Levels Completed'
    # target = 'last_campaign_started'
    num_labels = 2
    df, y = fix_target_and_drop_target_fields(df, target)
    print list(df.columns)
    y, name = categorize_by_level_num(y, num_labels)
    # y, name = categorize_by_level_num(y, num_labels, level_predict)
    # y, name = categorize_by_campaign(y, num_labels)

    features = list(df.columns)

    X = df.values
    # X = extreme_filter(X).values

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train, X_test = train_test_split(X)

    ''' Multinomial Logistic Regression '''
    '''
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    mlr_feature_importance = model_multi_log_reg(
        X_train_scaled, y_train, name, X_test, y_test)
    mlr_feature_importance_readable = rank_features(
        features, mlr_feature_importance)[::-1]
    print mlr_feature_importance_readable
    '''
    ''' Random Forest'''
    rf_feature_importance, proba = model_random_forest(
        X_train, y_train, name, X_test, y_test)

    rf_feature_importance_readable = rank_features(
        features, rf_feature_importance)[::-1]
    print rf_feature_importance_readable

    '''Random Model'''
    # compare_with_random_mlr_model(X_train_scaled, y_train)
    print "\n*************  Random Model  *****************"
    random_X_train = random_feature(X_train)
    random_X_test = random_feature(X_test)

    '''baseline model'''
    baseline_X_train = X_train[:, 2]
    baseline_X_test = X_test[:, 2]

    '''CHANGE THIS!'''

    # random_y_pred = model_random_forest(
    #     random_X_train, y_train, name, random_X_test, y_test)

    '''*** Results *** '''
    '''
    On TEST data:
    The Random Forest model categories by level with 2 classes yielded a model with:
        accuracy = 0.58696554587,
        precision = 0.671484157865,
        recall = 0.749379652605
        f1 - score = 0.708296687189
    '''
