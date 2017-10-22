import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_files(path):
    event_header = ['Created', 'User Id', 'Event Name', 'Level',
                    'raw_event1', 'raw_event2', 'raw_event3', 'raw_event4']

    df_users = pd.read_csv(path + 'users.csv')
    df_levels = pd.read_csv(path + 'levels.csv')
    df_events = pd.read_csv(path + 'events.csv', skiprows=1,
                            names=event_header, error_bad_lines=False, warn_bad_lines=True)
    return (df_users, df_levels, df_events)


def get_campaign_freq(df_levels):
    campaigns = pd.read_csv('../../data/campaign_list.csv')
    campaigns = campaigns.rename(index=str, columns={
                                 'Campaign': 'last_campaign_started', 'Level': 'level_name'})
    df_levels_freq = pd.DataFrame(
        data=df_levels['Level'].value_counts())
    df_levels_freq['level_name'] = df_levels_freq.index
    df_levels_freq = pd.merge(
        df_levels_freq, campaigns, how='left', on='level_name')
    return df_levels_freq


if __name__ == '__main__':
    sample_path = '../../data/sample/'
    august_path = '../../data/august/'
    march_path = '../../data/march/'

    path = march_path
    df_users, df_levels, df_events = read_files(path)

    df_levels_freq = get_campaign_freq(df_levels)
    # random selection of users: #42222 #4222 # 42
    # a few users with more than 20 levels: 19, 38, 39, 212
    user = 19
    user_df = df_users[df_users['Id'] == user]
    user_levels = df_levels[df_levels['User Id'] == user]
    user_events = df_events[df_events['User Id'] == user]
