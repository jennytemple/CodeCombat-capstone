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
    # level_predict_available = [10, 15, 30, 60, 100]
    level_predict_available = [15]
    for level_predict in level_predict_available:

        csv_name = 'model_predict_at_' + str(level_predict) + '.csv'

        df = pd.read_csv(path + csv_name)
        df = drop_unnamed(df)  # unnamed field when read in csv
        # drop users who completed < num_levels
        df = filter_missing(df)

        df, y = fix_target_and_drop_target_fields(df, 'Levels Completed')
        y, name = categorize_by_level_num(y, 2, threshold=level_predict)
        X = df.values
        df_toggle = df.copy()

        # df_toggle['Signed Up'] = False
        # df_toggle['Signed Up'] = True
        # df_toggle['avg_time_to_complete_level_chunk_5'] = df_toggle['avg_time_to_complete_level_chunk_5'] * 2.0
        df_toggle['avg_time_to_complete_level_chunk_5'] = df_toggle['avg_time_to_complete_level_chunk_5'] * .5
        # df_toggle['avg_time_to_complete_level_chunk_5'] = df_toggle['avg_time_to_complete_level_chunk_5'].mean()
        # df_toggle['avg_time_to_complete_level_chunk_10'] = df_toggle['avg_time_to_complete_level_chunk_10'] * 2
        # df_toggle['avg_time_to_complete_level_chunk_10'] = df_toggle['avg_time_to_complete_level_chunk_10'] * .5
        # df_toggle['avg_time_to_complete_level_chunk_10'] = df_toggle['avg_time_to_complete_level_chunk_10'].mean()
        # df_toggle['other_age'] = 1
        # df_toggle['13-15'] = 0
        # df_toggle['16-17'] = 0
        # df_toggle['18-24'] = 0
        # df_toggle['25-34'] = 0
        # df_toggle['35-44'] = 0
        # df_toggle['45-100'] = 0
        # df_toggle['rate_started_level_chunk_10'] = df_toggle['rate_started_level_chunk_10'] * 2
        # df_toggle['rate_started_level_chunk_10'] = 1.0

        # df_toggle['rate_started_level_chunk_60'] = df_toggle['rate_started_level_chunk_60'] * 2
        # df_toggle['rate_started_level_chunk_60'] = 1.0
        # df_toggle['rate_practice_levels_chunk_60'] = df_toggle['rate_practice_levels_chunk_60'] * 2.0
        # df_toggle['rate_practice_levels_chunk_60'] = df_toggle['rate_practice_levels_chunk_60'] * 0.5
        # df_toggle['rate_show_problem_alerts_chunk_30'] = df_toggle['rate_show_problem_alerts_chunk_30'] * 3.0
        # df_toggle['rate_show_problem_alerts_chunk_30'] = df_toggle['rate_show_problem_alerts_chunk_30'] * 0.1
        # df_toggle['avg_time_to_complete_level_chunk_30'] = df_toggle['avg_time_to_complete_level_chunk_30'] * 2
        # df_toggle['avg_time_to_complete_level_chunk_30'] = df_toggle['avg_time_to_complete_level_chunk_30'] * .5
        # df_toggle['avg_time_to_complete_level_chunk_30'] = df_toggle['avg_time_to_complete_level_chunk_30'].mean()
        # df_toggle['avg_time_to_complete_level_chunk_10'] = df_toggle['avg_time_to_complete_level_chunk_10'] * 5
        # df_toggle['avg_time_to_complete_level_chunk_10'] = df_toggle['avg_time_to_complete_level_chunk_10'] * .1
        # df_toggle['avg_time_to_complete_level_chunk_10'] = df_toggle['avg_time_to_complete_level_chunk_10'].mean()
        X_toggle = df_toggle.values

        # real Random Forest model
        rf_model_to_use = 'models/model_predict_' + \
            str(level_predict) + '_rf_grid.p'
        final_rf_model = pickle.load(open(rf_model_to_use, "rb"))

        y_predict_model = final_rf_model.predict(X)
        y_predict_toggle = final_rf_model.predict(X_toggle)
        print "Real data predicts {} will continue to play".format(y_predict_model.sum())
        print "Toggled data predicts {} will continue to play".format(y_predict_toggle.sum())
