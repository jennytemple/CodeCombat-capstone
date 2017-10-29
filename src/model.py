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


def filter_already_dropped(df, num_level):
    df = df[df['Levels Completed'] >= num_level]
    return df


def filter_missing(df):
    '''
    removes rows of data with any missing values
    note: missing values may be from the small number of users who deviate from the typical level progression. Data owners stated that ignoring these users may be best
    '''
    return df.dropna(axis=0, how='any')


def categorize_by_level_num(level_nums, num_cats, threshold=13):
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


def drop_countries(df):
    countries = ['argentina', 'australia', 'austria', 'belarus',
                 'belgium', 'brazil', 'bulgaria', 'canada',
                 'chile', 'colombia', 'croatia', 'czech-republic',
                 'denmark', 'ecuador', 'egypt', 'estonia',
                 'finland', 'germany', 'greece',
                 'hong-kong', 'hungary', 'india', 'indonesia',
                 'iran', 'ireland', 'israel', 'italy',
                 'japan', 'kazakhstan', 'lithuania', 'macedonia',
                 'malaysia', 'mexico', 'netherlands', 'new-zealand',
                 'norway', 'other_country', 'pakistan', 'peru',
                 'philippines', 'poland', 'portugal',
                 'romania', 'saudia-arabia', 'serbia', 'singapore',
                 'slovakia', 'slovenia', 'south-africa', 'south-korea',
                 'spain', 'sweden', 'switzerland', 'taiwan', 'thailand',
                 'turkey', 'ukraine', 'united-arab-emirates',
                 'venezuela', 'vietnam', 'united-kingdom', 'russia', 'france',
                 'united-states'
                 ]
    for c in countries:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
    return df


def drop_ages(df):
    ages = ['13-15',
            '16-17',
            '18-24',
            '25-34',
            '35-44',
            '45-100',
            'other_age']
    for age in ages:
        if age in df.columns:
            df.drop(age, axis=1, inplace=True)
    return df


'''functionality moved upstream '''
# def build_up_12(df):
#     df['pythonchunk_12'] = df['pythonchunk_12'] > .5
#
#     drop_these = [
#         # 'Signed Up',
#         # 'Paid Subscription',
#         # 'english_speaking',
#         # 'avg_time_to_complete_level_chunk_12'
#         # 'rate_hint_used_chunk_12',
#         # 'rate_hints_clicked_chunk_12',
#         # 'rate_hints_next_clicked_chunk_12',
#         # 'rate_started_level_chunk_12',
#         # 'rate_show_problem_alerts_chunk_12',
#         # 'rate_practice_levels_chunk_12',
#         'luachunk_12',
#         # 'pythonchunk_12',
#         'javachunk_12',
#         'javascriptchunk_12',
#         'coffeescriptchunk_12',
#         # 'logins_chunk_12'
#     ]
#     df.drop(drop_these, axis=1, inplace=True)
#     return df


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
    con_mat = confusion_matrix(y_true, y_pred)
    print con_mat
    con_mat_total = np.sum(con_mat) * 1.0
    print np.around(con_mat[0, 0] / con_mat_total, 2), np.around(con_mat[0, 1] / con_mat_total, 2)
    print np.around(con_mat[1, 0] / con_mat_total, 2), np.around(con_mat[1, 1] / con_mat_total, 2)


def model_multi_log_reg(X_train_scaled, y_train, name, X_test, y_test):
    '''
    Model with a multinomial logistic regression and print results
    '''
    multi_lr = LogisticRegression(
        multi_class='multinomial', solver='sag', max_iter=1000, class_weight='balanced')
    multi_lr.fit(X_train_scaled, y_train)

    # print "\nOn TRAIN data:"
    # y_predict = multi_lr.predict(X_train_scaled)
    # print "The Multinomial logistic regression modeled {} with {} classes yielded a model with:".format(name, num_labels)
    # print_scores(y_train, y_predict)
    #
    # print "\nOn TEST data:"
    # y_predict = multi_lr.predict(X_test)
    # print "The Multinomial logistic regression modeled {} with {} classes yielded a model with:".format(name, num_labels)
    # print_scores(y_test, y_predict)

    return multi_lr.coef_


def model_log_reg_lasso(X_train_scaled, y_train, name, X_test, y_test):
    '''
    Model with a multinomial logistic regression and print results
    '''
    lrl = LogisticRegression(penalty="l1", C=.5, solver='saga', max_iter=1000)
    lrl.fit(X_train_scaled, y_train)

    print "\nOn TRAIN data:"
    y_predict = lrl.predict(X_train_scaled)
    print "The Multinomial logistic regression modeled {} with {} classes yielded a model with:".format(name, num_labels)
    print_scores(y_train, y_predict)

    print "\nOn TEST data:"
    y_predict = lrl.predict(X_test)
    print "The Multinomial logistic regression modeled {} with {} classes yielded a model with:".format(name, num_labels)
    print_scores(y_test, y_predict)

    return lrl.coef_


def model_random_forest_grid(X_train, y_train, name, X_test, y_test, file_name="test"):
    '''
    Model with a random forest and print results
    '''
    # n_estimators, max_features="m", max_depth
    tuned_parameters = [
        {'max_features': ["sqrt", 0.3, 0.5, 0.7], 'min_samples_leaf': [1, 3, 5, 10]}]

    rand_forest = GridSearchCV(estimator=RandomForestClassifier(
        n_estimators=150), param_grid=tuned_parameters, scoring="accuracy", cv=5)  # verbose=100
    rand_forest.fit(X_train, y_train)
    print "\nGrid search score: {}".format(rand_forest.best_score_)
    print "\nThe best parameters:\n\tn_estimators = {}\n\tmax_features = {}\n\tmin_samples_leaf = {}"\
        .format(rand_forest.best_estimator_.n_estimators, rand_forest.best_estimator_.max_features, rand_forest.best_estimator_.min_samples_leaf)

    tuned_rf = RandomForestClassifier(n_estimators=1000,
                                      max_features=rand_forest.best_estimator_.max_features,
                                      min_samples_leaf=rand_forest.best_estimator_.min_samples_leaf
                                      )
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

    # fpr, tpr, thresholds = roc_curve(y_test, proba[:, 1])
    auc = roc_auc_score(y_test, proba[:, 1])
    print "\nThe area under the roc curve is: {}\n".format(auc)

    pickle.dump(tuned_rf, open(file_name + '.p', 'wb'))

    '''baseline model'''

    print "\n************** Baseline **************************\n"
    if np.sum(y) > len(y) / 2:
        baseline_y_test = np.ones(y_test.shape)
    else:
        baseline_y_test = np.zeros(y_test.shape)
    print_scores(y_test, baseline_y_test)
    return tuned_rf.feature_importances_, proba


def model_random_forest_no_grid(X_train, y_train, name, X_test, y_test, model_weight=None, file_name="test"):
    '''
    Model with a random forest and print results
    '''
    rand_forest = RandomForestClassifier(
        n_estimators=100, class_weight=model_weight)
    rand_forest.fit(X_train, y_train)

    print "\nOn TRAIN data:"
    y_predict = rand_forest.predict(X_train)
    print "The Random Forest model {} with {} classes yielded a model with:".format(name, num_labels)
    print_scores(y_train, y_predict)

    print "\nOn TEST data:"
    y_predict = rand_forest.predict(X_test)
    print "The Random Forest model {} with {} classes yielded a model with:".format(name, num_labels)
    print_scores(y_test, y_predict)
    proba = rand_forest.predict_proba(X_test)

    # fpr, tpr, thresholds = roc_curve(y_test, proba[:, 1])
    auc = roc_auc_score(y_test, proba[:, 1])
    print "\nThe area under the roc curve is: {}\n".format(auc)

    pickle.dump(rand_forest, open(file_name + '.p', 'wb'))

    '''baseline model'''

    print "\n************** Baseline **************************\n"
    if np.sum(y) > len(y) / 2:
        baseline_y_test = np.ones(y_test.shape)
    else:
        baseline_y_test = np.zeros(y_test.shape)
    print_scores(y_test, baseline_y_test)

    return rand_forest.feature_importances_, proba


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
    old_path = '~/galvanize/project/data/march/'

    path = march_path

    levels_available = [10, 15, 30, 60, 100]
    for level_predict in levels_available:
        csv_name = 'model_predict_at_' + str(level_predict) + '.csv'
        print "\n______________________________________"
        print "\t\t*************************"
        print "\n** Now building model to predict user churn at Level{} **".format(level_predict)
        print "\t\t*************************"

        df = pd.read_csv(path + csv_name)

        df = drop_unnamed(df)  # unnamed field when read in csv
        # df = drop_countries(df)  # too much noise
        # df = drop_ages(df)  # too much noise
        # df = build_up_12(df)  # keep fields that appear valuable to model
        # drop users who completed < num_levels
        df = filter_missing(df)

        target = 'Levels Completed'
        # target = 'last_campaign_started'
        num_labels = 2
        df, y = fix_target_and_drop_target_fields(df, target)
        print list(df.columns)
        y, name = categorize_by_level_num(
            y, num_labels, threshold=level_predict)
        # y, name = categorize_by_level_num(y, num_labels, level_predict)
        # y, name = categorize_by_campaign(y, num_labels)

        features = list(df.columns)

        X = df.values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y)  # for testing: random_state=42

        ''' Random Forest'''
        # rf_feature_importance, proba = model_random_forest_no_grid(
        #     X_train, y_train, name, X_test, y_test, model_weight=None, file_name="test4")  # None or "balanced"

        rf_feature_importance, proba = model_random_forest_grid(
            X_train, y_train, name, X_test, y_test, file_name="models/model_predict_" + str(level_predict) + "_rf_grid")

        rf_feature_importance_readable = rank_features(
            features, rf_feature_importance)[::-1]
        print "\nFeature importance from the Random Forest:"
        print rf_feature_importance_readable

        ''' Multinomial Logistic Regression '''
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)

        mlr_feature_importance = model_multi_log_reg(
            X_train_scaled, y_train, name, X_test, y_test)
        mlr_feature_importance_readable = rank_features(
            features, mlr_feature_importance)[::-1]
        print "\nFor genearl understanding of direction, feature importance from fitting a Logistic Regression model:"
        print mlr_feature_importance_readable
