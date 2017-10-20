import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


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

def categorize_by_level_num (level_nums, num_cats):

    '''
    For now, use number of levels, but later do by last campaign
    campaigns = pd.read_csv('../data/campaign_list.csv')
    general order of campaings: dungeon, campaign-web-dev-1, campaign-game-dev-1, forest, campaign-web-dev-2, campaign-game-dev-2, desert, mountain, glacier
    '''
    name = "categories by level"
    if num_cats == 2:
        bins = np.array([5, 20, 999])
    if num_cats == 4:
        bins = np.array([5, 12, 39, 100, 999])

    y = np.digitize(level_nums, bins)
    return y, name

def categorize_by_campaign (y, num_cats):

    '''
    Use to predict where in game the player will end instead of number of levels played
    campaigns = pd.read_csv('../data/campaign_list.csv')
    general order of campaigns: dungeon, campaign-web-dev-1, campaign-game-dev-1, forest, campaign-web-dev-2, campaign-game-dev-2, desert, mountain, glacier, other(can be anywhere)
    '''
    name = "categories by campaign"
    if num_cats == 2:
        early = ['dungeon']
        y = y.apply(lambda row: "early_churn" if row in ['dungeon'] else "later_churn")

    if num_cats == 3:
        early = ['dungeon']
        mid = ['campaign-web-dev-1', 'campaign-game-dev-1', 'forest']
        y = y.apply(lambda row: "early_churn" if row in early else ("mid_churn" if row in mid else "later_churn"))

    if num_cats == 4:
        early = ['dungeon']
        mid = ['campaign-web-dev-1', 'campaign-game-dev-1', 'forest']
        late = ['campaign-web-dev-2', 'campaign-game-dev-2', 'desert', 'mountain', 'glacier']
        y = y.apply(lambda row: "early_churn" if row in early else ("mid_churn" if row in mid else ("later_churn" if row in late else "other")))
    # if num_cats == 4:
    #     early = ['dungeon']
    #     mid = ['forest']
    #     late = ['campaign-web-dev-1', 'campaign-game-dev-1','campaign-web-dev-2', 'campaign-game-dev-2', 'desert', 'mountain', 'glacier']
    #     y = y.apply(lambda row: "early_churn" if row in early else ("mid_churn" if row in mid else ("later_churn" if row in late else "other")))
    #could also investigate campaign vs dev
    return y.values, name


def drop_fields(df):
    zero_this_data_set = ['hints_clicked_first_six', 'hints_used_first_six']
    captured_in_target = ['Playtime (s)', 'last_event_date', 'last_action', 'last_level_name', 'active_time_days', 'data_through', 'activity_gap_days', 'Level', 'last_level_time_s', 'daygap', 'was_completed', 'last_level_started']
    may_capture_target = ['avg_play_time_per_level_s', 'avg_num_days_per_level']
    not_useful = ['Unnamed: 0','Id', 'Date Joined', 'Practice']
    too_sparse = ['Gender','Want to be a programmer?', 'How long have you been programming?', 'How hard is CodeCombat?','How did you hear about CodeCombat?','Gender?', 'Favorite programming language?', 'Early bird or night owl?', 'What polls do you like?', 'Friends who code?', 'How fast is your internet?', 'After playing CodeCombat',' how interested are you in programming?', 'How likely are you to recommend CodeCombat?', "How likely that you'd recommend CodeCombat?"]
    df.drop(zero_this_data_set, axis=1, inplace=True)
    df.drop(captured_in_target, axis=1, inplace=True)
    df.drop(may_capture_target, axis=1, inplace=True)
    df.drop(not_useful, axis=1, inplace=True)
    df.drop(too_sparse, axis=1, inplace=True)

    return df


def drop_unused_labels(df):
    '''
    check columns that may have been dropped already with target choice and drop if still included
    '''
    if 'Levels Completed' in df.columns:
        df.drop('Levels Completed', axis=1, inplace=True)
    if 'last_campaign_started' in df.columns:
        df.drop('last_campaign_started', axis=1, inplace=True)

    return df


def dummify_with_countries(X):
    countries = pd.get_dummies(X['Country'])
    X[countries.columns] = countries

    languages = pd.get_dummies(X['Code Language'])
    X[languages.columns] = languages

    ages = pd.get_dummies(X['How old are you?'])
    X[ages.columns] = ages

    X.drop(['Country', 'Code Language','How old are you?'], axis=1, inplace=True)
    return X

def dummify_no_countries(X):
    languages = pd.get_dummies(X['Code Language'])
    X[languages.columns] = languages

    ages = pd.get_dummies(X['How old are you?'])
    X[ages.columns] = ages

    X.drop(['Country', 'Code Language', 'How old are you?'], axis=1, inplace=True)
    return X

def extreme_filter(X):
    '''
    Make a very simple version of X to test that model is running
    '''
    X = X[['avg_first_six_playtime_s']]
    return X

def random_feature(X):
    X = np.random.rand(X.shape[0],1)
    return X

def print_scores(y_true, y_pred, multi_lr_accuracy):
    multi_lr_f1_score_none = f1_score(y_true, y_pred, average=None)
    multi_lr_f1_score_macro = f1_score(y_true, y_pred, average='macro')
    multi_lr_f1_score_weighted = f1_score(y_true, y_pred, average='weighted')

    print "\taccuracy = {}\n\tF1 score for each class = {} \n\tF1 score, macro = {} \n\tF1 score, weighted {}".format(multi_lr_accuracy, multi_lr_f1_score_none, multi_lr_f1_score_macro, multi_lr_f1_score_weighted)

def model_multi_log_reg(X_train_scaled, y_train, name):
    '''
    Model with a multinomial logistic regression and print results
    '''
    #For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’
    multi_lr = LogisticRegression(multi_class='multinomial',solver='sag', max_iter=1000, class_weight='balanced')
    multi_lr.fit(X_train_scaled, y_train)

    multi_lr_acc = multi_lr.score(X_train_scaled, y_train)

    y_predict = multi_lr.predict(X_train_scaled)

    print "\nThe Multinomial logistic regression modeled {} with {} classes yielded a model with:".format(name, num_labels)
    print_scores(y_train, y_predict, multi_lr_acc)

    return y_predict
    #multi_lr.score(scaler.transform(X_test), y_test)

def compare_with_random_model(orig_X, y_train):
    X = random_feature(orig_X)
    rando_pred = model_multi_log_reg(X,y_train, "random feature")

if __name__=='__main__':

    turn_off_warnings()

    sample_path = '../../data/sample/'
    august_path = '../../data/august/'
    march_path = '../../data/march/'

    path = march_path
    df = pd.read_csv(path+'post_processed_users.csv')

    df = drop_fields(df)
    df = filter_missing(df)

    num_labels = 2
    y, name = categorize_by_level_num(df.pop('Levels Completed'),num_labels)
    #y, name = categorize_by_campaign(df.pop('last_campaign_started'), num_labels)

    df = drop_unused_labels(df)
    df = dummify_with_countries(df)
    #df = dummify_no_countries(df)

    X = df.values
    # X = extreme_filter(X).values

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    y_pred = model_multi_log_reg(X_train_scaled, y_train, name)
    #print "\nThe model makes {} predictions and the sum of predictions is {}".format(y_pred.shape[0], y_pred.sum())

    compare_with_random_model(X_train_scaled, y_train)

    '''*** Results *** '''
    '''
    using campaigns, 4
    The Multinomial logistic regression modeled categories by campaign with 4 classes yielded a model with:
	accuracy = 0.554455679456
	F1 score for each class = [ 0.7538172   0.06045137  0.25324543  0.04923077]
	F1 score, macro = 0.279186192289
	F1 score, weighted 0.62181023731

    The Multinomial logistic regression modeled random feature with 4 classes yielded a model with:
	accuracy = 0.604965979966
	F1 score for each class = [ 0.73497562  0.          0.25563354  0.        ]
	F1 score, macro = 0.247652291442
	F1 score, weighted 0.607271420649
    '''
    '''
    using num levels, 4
    The Multinomial logistic regression modeled categories by level with 4 classes yielded a model with:
	accuracy = 0.541461916462
	F1 score for each class = [ 0.73179918  0.37261799  0.09355247  0.05900846]
	F1 score, macro = 0.314244524829
	F1 score, weighted 0.570875200075

    The Multinomial logistic regression modeled random feature with 4 classes yielded a model with:
	accuracy = 0.174447174447
	F1 score for each class = [ 0.          0.41377939  0.          0.00800518]
	F1 score, macro = 0.105446142113
	F1 score, weighted 0.170672602754

    '''
    '''
    using campaigns, 2
    The Multinomial logistic regression modeled categories by campaign with 2 classes yielded a model with:
	accuracy = 0.683779058779 ********** best model
	F1 score for each class = [ 0.76560721  0.51417371]
	F1 score, macro = 0.639890464491
	F1 score, weighted 0.701204407814

    The Multinomial logistic regression modeled random feature with 2 classes yielded a model with:
	accuracy = 0.50025987526
	F1 score for each class = [ 0.59830228  0.33890677]
	F1 score, macro = 0.46860452851
	F1 score, weighted 0.531860067984
    '''
    '''
    using levels, 2
    The Multinomial logistic regression modeled categories by level with 2 classes yielded a model with:
	accuracy = 0.685031185031
	F1 score for each class = [ 0.79217459  0.34984882]
	F1 score, macro = 0.571011707769
	F1 score, weighted 0.748587821518

    The Multinomial logistic regression modeled random feature with 2 classes yielded a model with:
	accuracy = 0.90146002646
	F1 score for each class = [ 0.94817668  0.        ]
	F1 score, macro = 0.474088339442
	F1 score, weighted 0.854743374036
    '''
    '''
    using 3 campaigns
    The Multinomial logistic regression modeled categories by campaign with 3 classes yielded a model with:
	accuracy = 0.587507087507
	F1 score for each class = [ 0.76140716  0.09494192  0.3258348 ]
	F1 score, macro = 0.394061293252
	F1 score, weighted 0.644919767417

    The Multinomial logistic regression modeled random feature with 3 classes yielded a model with:
	accuracy = 0.402853902854
	F1 score for each class = [ 0.61799692  0.03861153  0.        ]
	F1 score, macro = 0.218869481722
	F1 score, weighted 0.46019777515
    '''
