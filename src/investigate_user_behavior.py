import numpy as np
import pandas as pd


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


def get_drop_off_by_num_levels(df_users):
    '''
    if percent_balk at Level 12 = .39, 39% of users who played 12 levels did not play more than 12 levels
    '''
    level_freq = df_users['Levels Completed'].value_counts()
    level_freq = pd.DataFrame(level_freq)
    level_freq['num_levels_completed'] = level_freq.index
    level_freq = level_freq.sort_values('num_levels_completed')
    level_freq = level_freq.rename(
        columns={'Levels Completed': 'final_num_users'})
    level_freq['total_users'] = level_freq['final_num_users'].sum()
    level_freq['cumulative_balked'] = level_freq['final_num_users'].cumsum()
    level_freq['users_remaining'] = level_freq['total_users'] - \
        level_freq['cumulative_balked']
    level_freq['users_previous'] = level_freq['users_remaining'].shift(1)
    level_freq['users_previous'] = level_freq['users_previous'].fillna(
        level_freq['total_users'])
    level_freq['percent_retained'] = level_freq['users_remaining'] / \
        level_freq['users_previous']
    level_freq['percent_balk'] = 1.0 - level_freq['percent_retained']

    # plot over time
    level_freq.plot.line('num_levels_completed', 'percent_balk')
    plt.show()

    # order by greatest % balk
    level_freq[['num_levels_completed', 'percent_balk']
               ].sort_values('percent_balk', ascending=False)
    return level_freq


def unpaid_users_over_level(df_users, df_levels, n=12):
    df_unpaid = df_users[df_users['Paid Subscription'] == False]
    df_unpaid = df_unpaid['Id'][df_unpaid['Levels Completed'] > n]
    levels_over_n_unpaid = df_levels[df_levels['User Id'].isin(df_unpaid)]
    return levels_over_n_unpaid


def unpaid_level_freq(df_unpaid):
    unpaid_level_freq = get_campaign_freq(df_unpaid)
    print unpaid_level_freq[unpaid_level_freq['last_campaign_started'] == 'dungeon']
    print unpaid_level_freq[unpaid_level_freq['last_campaign_started'] == 'forest']
    print unpaid_level_freq[unpaid_level_freq['last_campaign_started'] == 'desert']
    print unpaid_level_freq[unpaid_level_freq['last_campaign_started'] == 'mountain']
    print unpaid_level_freq[unpaid_level_freq['last_campaign_started'] == 'glacier']
    # results: forest starts at 13, but more dungeon is mixed in
    # desert starts solidly at 60 but other levels mixed in still
    # mountain is less clear 132,162, 165,
    # glacier has one level at 137, played by 193 people, but then no more until 345, and still not concentratetd

    # data to use: first dungeon set for


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

    level_freq = get_drop_off_by_num_levels(df_users)

    df_unpaid_over_12_levels = unpaid_users_over_level(
        df_users, df_levels, n=12)
    df_freq_unpaid_over_12_levels = get_campaign_freq(df_unpaid_over_12_levels)
