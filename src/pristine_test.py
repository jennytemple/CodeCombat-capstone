import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import pickle
from model import turn_off_warnings, drop_unnamed, filter_missing, categorize_by_level_num, fix_target_and_drop_target_fields, print_scores


if __name__ == '__main__':

    turn_off_warnings()
    march_path = '~/galvanize/project/data/march/test/'

    path = march_path
    level_predict_available = [10, 15, 30, 60, 100]
    # level_predict_available = [100]
    for level_predict in level_predict_available:

        csv_name = 'model_predict_at_' + str(level_predict) + '.csv'

        df = pd.read_csv(path + csv_name)
        df = drop_unnamed(df)  # unnamed field when read in csv
        # drop users who completed < num_levels
        df = filter_missing(df)

        df, y = fix_target_and_drop_target_fields(df, 'Levels Completed')
        y, name = categorize_by_level_num(y, 2, threshold=level_predict)
        X = df.values

        # real Random Forest model
        rf_model_to_use = 'models/model_predict_' + \
            str(level_predict) + '_rf_grid.p'
        final_rf_model = pickle.load(open(rf_model_to_use, "rb"))

        y_predict = final_rf_model.predict(X)
        proba = final_rf_model.predict_proba(X)

        print"\n\t******************************"
        print"\nUsing this model: {} to make prediction on this data: {}".format(rf_model_to_use, csv_name)
        print"\n\t******************************"
        print "\nThe results of the RANDOM FOREST model on pristine test data:"
        print_scores(y, y_predict)
        # fpr, tpr, thresholds = roc_curve(y_test, proba[:, 1])
        auc = roc_auc_score(y, proba[:, 1])
        print "\nThe area under the roc curve is: {}\n".format(auc)

        # real Lasso Logistic Regression model
        # scaler = pickle.load(open('models/feature_scaler.p', "rb"))
        # X_scaled = scaler.transform(X) #error: "operands could not be broadcast together"
        X_scaled = scale(X)
        lrl_model_to_use = 'models/model_predict_' + \
            str(level_predict) + '_lrl.p'
        final_lrl_model = pickle.load(open(lrl_model_to_use, "rb"))
        y_predict = final_lrl_model.predict(X_scaled)
        proba = final_lrl_model.predict_proba(X_scaled)

        print"\n\t******************************"
        print"\nUsing this model: {} to make prediction on this data: {}".format(lrl_model_to_use, csv_name)
        print"\n\t******************************"
        print "\nThe results of the LOGISTIC REGRESSION LASSO model on pristine test data:"
        print_scores(y, y_predict)
        # fpr, tpr, thresholds = roc_curve(y_test, proba[:, 1])
        auc = roc_auc_score(y, proba[:, 1])
        print "\nThe area under the roc curve is: {}\n".format(auc)

        print "\nThe results of the model on baseline (predicting the majority class):"
        if np.sum(y) > len(y) / 2:
            baseline_y = np.ones(y.shape)
        else:
            baseline_y = np.zeros(y.shape)

        print_scores(y, baseline_y)
