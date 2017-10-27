import pandas as pd
import numpy as np


def purge_to_merge(df_orig, num):
    # note:due to multiple merging it will be better to drop fields that aren't
    # used rather than keeping fields that are
    df = df_orig.copy()
    drop_these = ['Levels Completed',
                  'Signed Up',
                  'Paid Subscription',
                  'last_level_started',
                  'last_campaign_started',
                  'english_speaking',
                  'argentina', 'australia', 'austria', 'belarus',
                  'belgium', 'brazil', 'bulgaria', 'canada',
                  'chile', 'colombia', 'croatia', 'czech-republic',
                  'denmark', 'ecuador', 'egypt', 'estonia',
                  'finland', 'france', 'germany', 'greece',
                  'hong-kong', 'hungary', 'india', 'indonesia',
                  'iran', 'ireland', 'israel', 'italy',
                  'japan', 'kazakhstan', 'lithuania', 'macedonia',
                  'malaysia', 'mexico', 'netherlands', 'new-zealand',
                  'norway', 'other_country', 'pakistan', 'peru',
                  'philippines', 'poland', 'portugal',
                  'romania', 'russia', 'saudia-arabia', 'serbia', 'singapore',
                  'slovakia', 'slovenia', 'south-africa', 'south-korea',
                  'spain', 'sweden', 'switzerland', 'taiwan', 'thailand',
                  'turkey', 'ukraine', 'united-arab-emirates',
                  'united-kingdom', 'united-states', 'venezuela', 'vietnam',
                  '13-15', '16-17', '18-24', '25-34', '35-44', '45-100', 'other_age',
                  'luachunk_' + str(num),
                  'pythonchunk_' + str(num),
                  'javascriptchunk_' + str(num),
                  'coffeescriptchunk_' + str(num),
                  'javachunk_' + str(num),
                  'logins_chunk_' + str(num)]
    # keep_these = ['Id', 'rate_hint_used_chunk_60',
    #               'rate_hints_clicked_chunk_' + str(num),
    #               'rate_hints_next_clicked_chunk_' + str(num),
    #               'rate_started_level_chunk_' + str(num),
    #               'rate_show_problem_alerts_chunk_' + str(num),
    #               'rate_practice_levels_chunk_' + str(num)]

    for field in df.columns:
        if field in drop_these:
            df.drop(field, axis=1, inplace=True)
    return df


def clean_and_write(df, name, path):
    df.drop(['Id'], axis=1, inplace=True)
    df.to_csv(path + name + '.csv', index=False)


if __name__ == '__main__':
    read_path = '~/galvanize/project/data/march/train_to_merge/'
    write_path = '~/galvanize/project/data/march/train/'

    df_l5 = pd.read_csv(read_path + 'Model_predict_at_10_users.csv')
    df_l10 = pd.read_csv(read_path + 'Model_predict_at_15_users.csv')
    df_l15 = pd.read_csv(read_path + 'Model_predict_at_30_users.csv')
    df_l30 = pd.read_csv(read_path + 'Model_predict_at_60_users.csv')
    df_l60 = pd.read_csv(read_path + 'Model_predict_at_100_users.csv')

    p_df_l5 = purge_to_merge(df_l5, 5)
    df_l10 = pd.merge(df_l10, p_df_l5, how='left', on='Id')

    p_df_l10 = purge_to_merge(df_l10, 10)
    df_l15 = pd.merge(df_l15, p_df_l10, how='left', on='Id')

    p_df_l15 = purge_to_merge(df_l15, 15)
    df_l30 = pd.merge(df_l30, p_df_l15, how='left', on='Id')

    p_df_l30 = purge_to_merge(df_l30, 30)
    df_l60 = pd.merge(df_l60, p_df_l30, how='left', on='Id')

    clean_and_write(df_l5, 'model_predict_at_10', write_path)
    clean_and_write(df_l10, 'model_predict_at_15', write_path)
    clean_and_write(df_l15, 'model_predict_at_30', write_path)
    clean_and_write(df_l30, 'model_predict_at_60', write_path)
    clean_and_write(df_l60, 'model_predict_at_100', write_path)
