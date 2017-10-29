import pandas as pd
import numpy as np


def general_cleanup(df_orig, num):
    # methodology changes to features made after producing initial data sets
    df = df_orig.copy()
    drop_these = ['luachunk_' + str(num),
                  'javascriptchunk_' + str(num),
                  'coffeescriptchunk_' + str(num),
                  'javachunk_' + str(num)]

    for field in df.columns:
        if field in drop_these:
            df.drop(field, axis=1, inplace=True)

    # make this binary:
    df['pythonchunk_' + str(num)] = df['pythonchunk_' + str(num)] > .5
    df = df.drop_duplicates()
    return df


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

    for field in df.columns:
        if field in drop_these:
            df.drop(field, axis=1, inplace=True)
    return df


def clean_and_write(df, name, path):
    df.drop(['Id'], axis=1, inplace=True)
    print df.shape
    df.to_csv(path + name + '.csv', index=False)


if __name__ == '__main__':
    read_path = '~/galvanize/project/data/march/train_to_merge/'
    write_path = '~/galvanize/project/data/march/train/'

    # read_path = '~/galvanize/project/data/march/test_to_merge/'
    # write_path = '~/galvanize/project/data/march/test/'

    # Prediction at 13 is a standalone file, not part of the model collection
    df = pd.read_csv(read_path + 'Model_predict_at_13_users.csv')
    # df = general_cleanup(df, 13)
    clean_and_write(df, 'model_predict_at_13', write_path)

    df_L05 = pd.read_csv(read_path + 'Model_predict_at_10_users.csv')
    df_L10 = pd.read_csv(read_path + 'Model_predict_at_15_users.csv')
    df_L15 = pd.read_csv(read_path + 'Model_predict_at_30_users.csv')
    df_L30 = pd.read_csv(read_path + 'Model_predict_at_60_users.csv')
    df_L60 = pd.read_csv(read_path + 'Model_predict_at_100_users.csv')

    df_L05 = general_cleanup(df_L05, 5)
    df_L10 = general_cleanup(df_L10, 10)
    df_L15 = general_cleanup(df_L15, 15)
    df_L30 = general_cleanup(df_L30, 30)
    df_L60 = general_cleanup(df_L60, 60)

    p_df_L05 = purge_to_merge(df_L05, 5)
    df_L10 = df_L10.merge(p_df_L05, how='left', on='Id')

    p_df_L10 = purge_to_merge(df_L10, 10)
    df_L15 = df_L15.merge(p_df_L10, how='left', on='Id')

    p_df_L15 = purge_to_merge(df_L15, 15)
    df_L30 = df_L30.merge(p_df_L15, how='left', on='Id')

    p_df_L30 = purge_to_merge(df_L30, 30)

    df_L60 = pd.merge(df_L60, p_df_L30, how='left', on='Id')
    print df_L60.shape

    clean_and_write(df_L05, 'model_predict_at_10', write_path)
    clean_and_write(df_L10, 'model_predict_at_15', write_path)
    clean_and_write(df_L15, 'model_predict_at_30', write_path)
    clean_and_write(df_L30, 'model_predict_at_60', write_path)
    clean_and_write(df_L60, 'model_predict_at_100', write_path)
