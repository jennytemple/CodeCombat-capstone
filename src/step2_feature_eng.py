import numpy as np
import pandas as pd

# from investigate_user_behavior import get_campaign_freq, unpaid_users_over_level


def read_files(path, t):
    event_header = ['Created', 'User Id', 'Event Name', 'Level',
                    'raw_event1', 'raw_event2', 'raw_event3', 'raw_event4']

    df_users = pd.read_csv(path + t + 'users.csv')
    if 'Unnamed: 0' in df_users.columns:
        df_users.drop('Unnamed: 0', axis=1, inplace=True)
    df_levels = pd.read_csv(path + 'levels.csv')

    df_events = pd.read_csv(path + 'events.csv', skiprows=1,
                            names=event_header, error_bad_lines=False, warn_bad_lines=True)
    # df_events_a = pd.read_csv(path + 'events_a.csv')
    # df_events_b = pd.read_csv(path + 'events_b.csv')
    # df_events = pd.concat([df_events_a, df_events_b], axis=0)

    return (df_users, df_levels, df_events)


def cleanup_events(df_events):
    df_events.drop(['raw_event1', 'raw_event2', 'raw_event3',
                    'raw_event4'], axis=1, inplace=True)
    df_events['Created'] = pd.to_datetime(df_events['Created'])
    if 'Unnamed: 0' in df_events.columns:
        df_events.drop('Unnamed: 0', axis=1, inplace=True)
    return df_events


def cleanup_levels(df_levels):
    df_levels['Created'] = pd.to_datetime(df_levels['Created'])
    if 'Unnamed: 0' in df_levels.columns:
        df_levels.drop('Unnamed: 0', axis=1, inplace=True)
    return df_levels


def add_level_num(df_levels):
    # df_levels = df_levels.sort_values('Created')  # ensure order is correct
    df_levels['level_num'] = df_levels.groupby('User Id').cumcount() + 1
    return df_levels


def add_event_num(df_l, df_events):
    df_levels = df_l.copy()
    df_levels = df_levels[['User Id', 'Level', 'level_num']]
    df_events = df_events.merge(df_levels, how='left',
                                on=['User Id', 'Level'])
    return df_events


def cleanup_users(df_users):
    df_users['Date Joined'] = pd.to_datetime(df_users['Date Joined'])
    # funky comma in csv headings. Move everything over to ac-coma-date
    df_users["How likely that you'd recommend CodeCombat?"] = df_users["How likely are you to recommend CodeCombat?"]
    df_users["How likely are you to recommend CodeCombat?"] = df_users[" how interested are you in programming?"]
    df_users["How interested are you in programming?"] = df_users["After playing CodeCombat"]
    df_users.drop([" how interested are you in programming?",
                   "After playing CodeCombat"], axis=1, inplace=True)
    return df_users


def user_filter(df_users, min_levels):
    df_users = df_users[df_users['Levels Completed'] >= min_levels]
    return df_users


def add_total_play_time_per_user(df_users, df_levels):
    '''get total time spent playing per user and add to df'''
    total_play_time = df_levels[['Playtime (s)', 'User Id']].groupby(
        'User Id').sum().sort_values(by='Playtime (s)', ascending=False)
    total_play_time = total_play_time.rename(
        index=str, columns={'Playtime (s)': 'total_play_time'})
    # reset index
    total_play_time['Id'] = total_play_time.index

    # merge files
    df_users = df_users.merge(total_play_time, how='left', on='Id')

    return df_users


def add_last_play_info(df_users, df_events):
    '''add information on last event - date, last action, and last level name'''

    # find latest event from event file
    df_last_event = df_events[['User Id', 'Created']
                              ].groupby(['User Id']).max()
    df_last_event['User Id'] = df_last_event.index  # reset index

    # merge other data fields on to last_event
    df_last_event = df_last_event.merge(df_events,
                                        how='left', on='User Id')
    # filter the resulting table for only the last events (but now with more information on the last events)
    df_last_event = df_last_event[df_last_event['Created_x']
                                  == df_last_event['Created_y']]
    # drop unnecessaryfields
    df_last_event.drop(['Created_y', 'level_num'], axis=1, inplace=True)

    # dedup
    df_last_event = df_last_event.drop_duplicates()
    # rename fields
    df_last_event = df_last_event.rename(index=str, columns={
                                         'Created_x': 'last_event_date', 'User Id': 'Id', 'Level': 'last_level_name', 'Event Name': 'last_action'})

    # issue: if multiple event types occur at exactly the same time, both will be in the data set

    # join information to users data frame
    df_users = df_users.merge(df_last_event, how='left', on='Id')
    return df_users


def add_active_days(df_users):
    '''add duration of membership in data set (creation date to last event date)'''
    # number of days between account creation and last action
    df_users['active_time_days'] = (pd.to_datetime(
        df_users['last_event_date']) - pd.to_datetime(df_users['Date Joined'])).dt.days
    # count the first day as active
    df_users['active_time_days'] = df_users['active_time_days'] + 1
    return df_users


def add_activity_gap_info(df_users, data_pull_date):
    '''add number of days between last member activity and when data set was pulled'''
    df_users['data_through'] = pd.to_datetime(data_pull_date)
    df_users['activity_gap_days'] = (
        df_users['data_through'] - pd.to_datetime(df_users['last_event_date'])).dt.days
    return df_users


def add_last_level_completion_info(df_users, df_levels):
    '''add field for if user last completed a level or was last mid-level'''
    # select last event
    df_last_level = df_levels[['User Id', 'Created']
                              ].groupby(['User Id']).max()
    # reset index
    df_last_level['User Id'] = df_last_level.index
    # merge rest of data on last event date
    df_last_level = df_last_level.merge(df_levels,
                                        how='left', on='User Id')
    # select only the last event (but now with detail)
    df_last_level = df_last_level[df_last_level['Created_x']
                                  == df_last_level['Created_y']]
    # add indicator for if the last level was completed
    df_last_level['last_level_was_completed'] = df_last_level['Date Completed'] > 0
    # drop unnecesary fields
    df_last_level.drop(
        ['Created_y', 'Created_x', 'Date Completed', 'Code Language', 'level_num'], axis=1, inplace=True)
    # rename columns
    df_last_level = df_last_level.rename(
        index=str, columns={'User Id': 'Id', 'Playtime (s)': 'last_level_time_s', 'Practice': 'last_level_was_practice', 'Level': 'last_level_played'})

    # merge level data on to user data
    df_users = df_users.merge(df_last_level, how='left', on='Id')
    return df_users


def add_avg_play_per_level(df_users):
    '''add the average play time per level for all levels played'''
    df_users['avg_play_time_per_level_s'] = df_users['total_play_time'] / \
        df_users['Levels Completed']
    return df_users


def add_avg_days_per_level(df_users):
    '''add the average levels/day for all levels played '''
    df_users['daygap'] = (pd.to_datetime(
        df_users['last_event_date']) - pd.to_datetime(df_users['Date Joined'])).dt.days + 1
    df_users['avg_num_days_per_level'] = df_users['daygap'] / \
        df_users['Levels Completed']
    return df_users


def add_last_level_started(df_users, df_l):
    df_levels = df_l.copy()
    df_levels = df_levels.dropna(axis=0, how='any')
    df_last_level_started = df_levels[['User Id', 'Created']].groupby([
                                                                      'User Id']).max()
    df_last_level_started['User Id'] = df_last_level_started.index
    df_last_level_started = df_last_level_started.merge(
        df_levels, how='left', on='User Id')
    # select only the last event (but now with detail)
    df_last_level_started = df_last_level_started[df_last_level_started['Created_x']
                                                  == df_last_level_started['Created_y']]
    df_last_level_started['last_level_started'] = df_last_level_started['Level']
    df_last_level_started['Id'] = df_last_level_started['User Id']
    df_last_level_started = df_last_level_started[['Id', 'last_level_started']]
    df_last_level_started = df_last_level_started.drop_duplicates()
    df_users = df_users.merge(df_last_level_started, how='left', on='Id')
    return df_users


def add_last_campaign_started(df_users, aws=False):
    if aws == True:
        campaigns = pd.read_csv('data/campaign_list.csv')
    else:
        campaigns = pd.read_csv('../../data/campaign_list.csv')
    campaigns = campaigns.rename(index=str, columns={
                                 'Campaign': 'last_campaign_started', 'Level': 'last_level_started'})
    df_users = df_users.merge(campaigns, how='left',
                              on='last_level_started')
    df_users['last_campaign_started'] = df_users['last_campaign_started'].fillna(
        'other')
    return df_users


def add_group_start_and_complete_date(df_users, df_e, level_group, name, byname=True):
    # filter events for level completions of groups
    df_event = df_e.copy()
    if byname == True:
        df_events_group = df_events[df_events['Level'].isin(level_group)]
    else:
        df_events_group = df_events[df_events['level_num'].isin(level_group)]
    df_events_group = df_events_group[df_events_group['Event Name']
                                      == 'Saw Victory']

    # take max date and add to dataframe
    df_events_group_max = df_events_group[[
        'User Id', 'Created']].groupby('User Id').max()
    df_events_group_max['Id'] = df_events_group_max.index
    df_events_group_max['date_completed_' +
                        name] = pd.to_datetime(df_events_group_max['Created'])
    df_events_group_max.drop(['Created'], axis=1, inplace=True)
    df_users = df_users.merge(df_events_group_max, how='left', on='Id')

    # take min date and add to dataframe
    df_events_group_min = df_events_group[[
        'User Id', 'Created']].groupby('User Id').min()
    df_events_group_min['Id'] = df_events_group_min.index
    df_events_group_min['date_started_' +
                        name] = pd.to_datetime(df_events_group_min['Created'])
    df_events_group_min.drop(['Created'], axis=1, inplace=True)
    # merge on user data
    df_users = df_users.merge(df_events_group_min, how='left', on='Id')
    return df_users


def add_group_completion_num(df_users, df_e, level_group, name, byname=True):
    # filter events for level completions of groups
    df_events = df_e.copy()
    if byname == True:
        df_events_group = df_events[df_events['Level'].isin(level_group)]
    else:
        df_events_group = df_events[df_events['level_num'].isin(level_group)]
    df_events_group = df_events_group[df_events_group['Event Name']
                                      == 'Saw Victory']
    # take max date and add to dataframe
    df_events_group_count = df_events_group[[
        'User Id', 'Event Name']].groupby('User Id').count()
    df_events_group_count['Id'] = df_events_group_count.index
    df_events_group_count = df_events_group_count.rename(
        index=str, columns={'Event Name': 'num_levels_completed_in_' + name})

    # merge on user data
    df_users = df_users.merge(df_events_group_count, how='left', on='Id')
    return df_users


def add_group_play_time(df_users, name):
    df_users['avg_time_to_complete_level_' + name] = (
        df_users['date_completed_' + name] - df_users['date_started_' + name]).dt.seconds / df_users['num_levels_completed_in_' + name]
    return df_users


def add_number_special_activities(df_users, df_e, event_type, earliest_date_field, latest_date_field, ref_field, new_field_name, use_events=True):
    '''count the number of times per user a specific event type occured within a time frame defined earlier and add it to the users df '''
    '''primarily built for event dataframe events, but also handles practice field in the level data frame'''
    '''Note: most special events are not tied to a particular level, so time frame has to be used instead of level '''
    df_events = df_e.copy()
    if use_events:
        df_events_special = df_events[df_events['Event Name'] == event_type]
    else:
        df_events_special = df_events[df_events['Practice'] == True]

    df_events_special = df_events_special.rename(
        index=str, columns={'User Id': 'Id', 'Created': 'event_date'})
    df_users_max_dates = df_users[[
        'Id', earliest_date_field, latest_date_field]]

    df_events_special = df_events_special.merge(
        df_users_max_dates, how='left', on='Id')
    df_events_special = df_events_special[(df_events_special['event_date'] >=
                                           df_events_special[earliest_date_field]) &
                                          (df_events_special['event_date'] <= df_events_special[latest_date_field])]

    event_count = df_events_special.groupby('Id').count()
    if use_events:
        event_count = event_count.rename(
            index=str, columns={'Event Name': new_field_name})
    else:
        event_count = event_count.rename(
            index=str, columns={'Practice': new_field_name})

    event_count = event_count[[new_field_name]]
    event_count['Id'] = event_count.index.astype(int)

    # merge and fill Nans with zeros
    df_users = df_users.merge(event_count, how='left', on='Id')
    df_users[new_field_name] = df_users[new_field_name].fillna(0)
    df_users[new_field_name] = df_users[new_field_name] / \
        df_users[ref_field]

    return df_users


def fill_out_age(df_users):
    df_users['How old are you?'] = df_users['How old are you?'].fillna(
        'other_age')
    return df_users


def add_country_group(df_users):
    us = ['united-states']
    other_english = ['united-kingdom', 'australia',
                     'canada', 'new-zealand', 'ireland', 'singapore']
    df_users['country_us'] = df_users['Country'].isin(us)
    df_users['country_english_ex_us'] = df_users['Country'].isin(other_english)
    df_users['country_non_english'] = - \
        (df_users['country_us'] | df_users['country_english_ex_us'])
    return df_users


def add_english_speaking(df_users):
    english = ['united-states', 'united-kingdom', 'australia',
               'canada', 'new-zealand', 'ireland', 'singapore']
    df_users['english_speaking'] = df_users['Country'].isin(english)
    return df_users


def aggregate_small_pop_countries(df_users, n=300):
    '''Lump countries with fewer than n users registering in data set as "other"
        note: choosing 300 for the March sample results in  5% of the population in the in "other"'''
    # countries = pd.DataFrame(df_users['Country'].value_counts())
    # countries_included = countries[countries['Country'] > n]
    # include = list(countries_included.index)
    # include = ['united-states', 'united-kingdom', 'australia', 'canada', 'germany', 'russia', 'france', 'spain', 'mexico', 'poland', 'taiwan', 'turkey', 'ukraine', 'colombia', 'south-korea', 'netherlands', 'new-zealand', 'japan', 'finland', 'brazil', 'hong-kong', 'norway', 'thailand', 'india', 'sweden', 'united-arab-emirates', 'denmark', 'singapore', 'italy', 'vietnam', 'hungary',
    #            'belgium', 'malaysia', 'austria', 'argentina', 'czech-republic', 'israel', 'indonesia', 'lithuania', 'romania', 'portugal', 'philippines', 'greece', 'south-africa', 'peru', 'switzerland', 'belarus', 'venezuela', 'slovakia', 'ireland', 'estonia', 'chile', 'serbia', 'slovenia', 'saudia-arabia', 'kazakhstan', 'bulgaria', 'ecuador', 'egypt', 'croatia', 'iran', 'pakistan', 'macedonia']
    include = ['united-states', 'united-kingdom',
               'australia', 'canada', 'germany', 'russia']
    df_users['Country'] = df_users.apply(
        lambda row: row['Country'] if row['Country'] in include else "other_country", axis=1)

    return df_users


def dummify_countries(df_users):
    countries = pd.get_dummies(df_users['Country'])
    df_users[countries.columns] = countries

    df_users.drop(['Country'], axis=1, inplace=True)
    return df_users


def dummify_ages(df_users):
    ages = pd.get_dummies(df_users['How old are you?'])
    df_users[ages.columns] = ages

    df_users.drop(['How old are you?'], axis=1, inplace=True)
    return df_users


def drop_unmodeled_fields(df):

    # dates = ['Date Joined']
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

    # df.drop(dates, axis=1, inplace=True)
    df.drop(eda_only_fields, axis=1, inplace=True)
    df.drop(too_sparse, axis=1, inplace=True)

    return df


def add_coding_language(df_users, df_l, name, level_group, num_levels_col, byname=True):
    '''Note: final column added may be greater than 1.0. Some levels are "completed" without seeing victory. Look into "the-raised-sword" to see detail'''
    df_levels = df_l.copy()
    df_user_num_levels = df_users[['Id', num_levels_col]]

    # include only completed levels
    df_levels = df_levels[df_levels['Date Completed'].notnull()]
    if byname:
        df_levels = df_levels[df_levels['Level'].isin(
            level_group)]  # include only levels in list
    else:
        df_levels = df_levels[df_levels['level_num'] <=
                              level_group]  # include only levels in list

    languages = list(df_levels['Code Language'].unique())
    for l in languages:
        df_levels_l = df_levels[df_levels['Code Language'] == l]  # filter
        df_levels_l = df_levels_l[['User Id', 'Code Language']].groupby(
            'User Id').count()
        df_levels_l['Id'] = df_levels_l.index
        df_levels_l = df_levels_l.merge(
            df_user_num_levels, how='left', on='Id')
        df_levels_l[l + name] = df_levels_l['Code Language'] / \
            df_levels_l[num_levels_col]

        # df_levels_l[l + name] = df_levels_l[l + name] >= 0.5

        df_levels_l.drop(['Code Language', num_levels_col],
                         axis=1, inplace=True)
        df_users = df_users.merge(df_levels_l, how='left', on='Id')
        df_users[l + name] = df_users[l + name].fillna(0)

    return df_users, languages


def add_number_logins(df_users, df_e, name, min_date_col, max_date_col, time_threshold_hours=1.0):
    '''for each user, and each event, add time between events, filter on max date, and count the number of events over a certain time '''
    # decrease size of df_events
    df_events = df_e.copy()
    df_events = df_events[['User Id', 'Created']]
    # filter dates
    df_users_max_dates = df_users[['Id', min_date_col, max_date_col]]
    df_users_max_dates = df_users_max_dates.rename(
        index=str, columns={'Id': 'User Id'})
    df_events = df_events.merge(df_users_max_dates,
                                how='left', on='User Id')
    # df_events = df_events[(df_events['Created'] >= df_events[min_date_col]) & (df_events['Created'] <= df_events[max_date_col])]

    # in two steps:
    df_events = df_events[df_events['Created'] >= df_events[min_date_col]]
    df_events = df_events[df_events['Created'] <= df_events[max_date_col]]

    df_events.drop([min_date_col, max_date_col], axis=1, inplace=True)
    # get time between events in hours
    # df_events = df_events.sort_values(
    #     'Created')  # data set appears to be sorted by time already, but repeat in case that assumption is incorrect
    df_events['shifted'] = df_events.groupby('User Id')['Created'].shift(1)
    # df_events['lag'] = df_events['Created'] - df_events['shifted']
    # df_events['lag_days'] = df_events['lag'].dt.days
    # df_events['lag_secs'] = df_events['lag'].dt.seconds
    # df_events['lag_hours'] = (df_events['lag_days'] * 24.0) + \
    #     (df_events['lag_secs'] / 60 / 60)
    df_events['lag_hours'] = (df_events['Created'] - df_events['shifted']).dt.days * \
        24.0 + (df_events['Created'] - df_events['shifted']
                ).dt.seconds / 60.0 / 60.0

    # get number of events exceeding time threshold
    df_events = df_events[df_events['lag_hours'] >= time_threshold_hours]
    df_events = df_events[['User Id', 'lag_hours']].groupby('User Id').count()
    df_events['Id'] = df_events.index

    df_users = df_users.merge(df_events, how='left', on='Id')
    df_users['logins_' + name] = df_users['lag_hours'].fillna(0)
    df_users.drop('lag_hours', axis=1, inplace=True)
    return df_users


def chunk_up_dataset2(choice):
    d = {}
    if choice == 1:
        d[5] = range(5)
    elif choice == 2:
        d[10] = range(5, 10)
    elif choice == 3:
        d[15] = range(10, 15)
    elif choice == 4:
        d[30] = range(15, 30)
    elif choice == 5:
        d[60] = range(30, 60)
    elif choice == 20:
        d[6] = range(6)
    elif choice == 30:
        d[12] = range(12)
    return d


def make_model_dict(choice):
    d = {}
    if choice == 1:
        d['Model_predict_at_10'] = [0, 5]
        l = ['Model_predict_at_10']
    elif choice == 2:
        d['Model_predict_at_15'] = [5, 10]
        l = ['Model_predict_at_15']
    elif choice == 3:
        d['Model_predict_at_30'] = [10, 15]
        l = ['Model_predict_at_30']
    elif choice == 4:
        d['Model_predict_at_60'] = [15, 30]
        l = ['Model_predict_at_60']
    elif choice == 5:
        d['Model_predict_at_100'] = [30, 60]
        l = ['Model_predict_at_100']

    elif choice == 20:
        d['Model_predict_at_13'] = [0, 6]
        l = ['Model_predict_at_13']

    elif choice == 30:  # different methodology for predicting 13
        d['Model_predict_at_13'] = [0, 12]
        l = ['Model_predict_at_13']

    return d, l


if __name__ == '__main__':
    sample_path = '../../data/sample/'
    august_path = '../../data/august/'
    march_path = '../../data/march/'
    tiny_sample_path = '../../data/tiny_sample/'

    # aws_march_path = '/home/ec2-user/data/'
    path = march_path
    df_users, df_levels, df_events = read_files(path, 'test/')
    print df_users.shape

    # df_users, df_levels, df_events = read_files(path, t='')

    # clean up and filter user df if necessary
    df_users = cleanup_users(df_users)
    print df_users.shape
    df_levels = cleanup_levels(df_levels)
    df_levels = add_level_num(df_levels)

    df_events = cleanup_events(df_events)
    df_events = add_event_num(df_levels, df_events)

    min_level_reached = 1  # 7
    df_users = user_filter(df_users, min_level_reached)
    print df_users.shape
    orig_fields = set(df_users.columns)

    # add fields to user df for eda (not to be used in model)
    df_users = add_total_play_time_per_user(
        df_users, df_levels)  # add play time
    print df_users.shape
    df_users = add_last_play_info(df_users, df_events)
    print df_users.shape
    df_users = add_active_days(df_users)
    print df_users.shape
    df_users = add_activity_gap_info(df_users, '2017-10-15')
    print df_users.shape
    df_users = add_last_level_completion_info(df_users, df_levels)
    print df_users.shape
    df_users = add_avg_play_per_level(df_users)
    print df_users.shape
    df_users = add_avg_days_per_level(df_users)
    added_eda_only_fields = set(df_users.columns) - orig_fields

    # add fields to user df for target modeling and eda
    df_users = add_last_level_started(df_users, df_levels)
    print df_users.shape
    df_users = add_last_campaign_started(df_users, aws=False)
    print df_users.shape
    added_target_fields = set(df_users.columns) - \
        orig_fields - added_eda_only_fields

    # add fields to user df for features modeling and eda
    df_users = fill_out_age(df_users)
    print df_users.shape
    # df_users = add_country_group(df_users)
    df_users = add_english_speaking(df_users)
    print df_users.shape
    df_users = aggregate_small_pop_countries(df_users)
    print df_users.shape
    df_users = dummify_countries(df_users)
    print df_users.shape
    df_users = dummify_ages(df_users)

    df_users = drop_unmodeled_fields(df_users)
    print df_users.shape
    added_modeling_fields = set(df_users.columns) - orig_fields - \
        added_eda_only_fields - added_target_fields

    # loop through number of models
    # set list_d to be different for each model
    # filter data set on num levels completed
    # write out df_users to different files for different models

    special_actions = ['Hint Used', 'Hints Clicked',
                       'Hints Next Clicked', 'Started Level', 'Show problem alert']
    special_names = ['rate_hint_used_', 'rate_hints_clicked_',
                     'rate_hints_next_clicked_', 'rate_started_level_', 'rate_show_problem_alerts_']

    choice = 5
    d = chunk_up_dataset2(choice=choice)
    model_dict, model_list = make_model_dict(choice=choice)

    for model_name in model_list:
        arr_d = np.sort(np.array(d.keys()))
        arr_d = arr_d[arr_d > model_dict[model_name][0]]
        arr_d = arr_d[arr_d <= model_dict[model_name][1]]

        df_users = df_users[df_users['Levels Completed'] >= arr_d[-1]]
        print df_users.shape
        print model_name, arr_d
        for i in arr_d:
            name = "chunk_" + str(i)
            group = d[i]
            df_users = add_group_start_and_complete_date(
                df_users, df_events, group, name, byname=False)
            df_users = add_group_completion_num(
                df_users, df_events, group, name, byname=False)

            '''new'''
            df_users = add_group_play_time(df_users, name)

            for j in xrange(len(special_actions)):
                df_users = add_number_special_activities(
                    df_users, df_events, special_actions[j], 'date_started_' + name, 'date_completed_' + name, 'num_levels_completed_in_' + name, special_names[j] + name)

            df_users = add_number_special_activities(
                df_users, df_levels, 'Practice', 'date_started_' + name, 'date_completed_' + name, 'num_levels_completed_in_' + name, 'rate_practice_levels_' + name, use_events=False)
            print df_users.shape
            # df_users = add_number_logins(
            #     df_users, df_events, name, 'date_started_' + name, 'date_completed_' + name, time_threshold_hours=1.0)
            # keep the last one in the set
            df_users['num_levels_completed_in'] = df_users['num_levels_completed_in_' + name]
            df_users['date_completed'] = df_users['date_completed_' + name]
            df_users.drop(['date_started_' + name, 'date_completed_' + name,
                           'num_levels_completed_in_' + name], axis=1, inplace=True)

        # add for entire time period only:
        df_users, language_list = add_coding_language(
            df_users, df_levels, name, i, 'num_levels_completed_in', byname=False)
        print df_users.shape
        df_users = add_number_logins(
            df_users, df_events, name, 'Date Joined', 'date_completed', time_threshold_hours=1.0)
        if 'Unnamed: 0' in df_users.columns:
            df_users.drop(['Unnamed: 0'], axis=1, inplace=True)
        print df_users.shape

        # write out csv file for later use
        df_output = df_users.drop(
            ['Date Joined', 'num_levels_completed_in', 'date_completed'], axis=1)
        df_output.to_csv(path + 'test_to_merge/' + model_name +
                         '_users.csv', index=False)
        print df_output.shape
        # drop fields not desired for next output csv
        df_users.drop(['logins_' + name], axis=1, inplace=True)
        for lang in language_list:
            df_users.drop([lang + name], axis=1, inplace=True)

        df_test = pd.read_csv(path + 'test_to_merge/' + model_name +
                              '_users.csv')
        print df_test.shape
    print "\n*********** CHOICE = {} ***************\n".format(choice)
    print list(df_output.columns)
    # print "Orig fields are:"
    # print list(orig_fields)
    # print "Fields added for EDA are:"
    # print list(added_eda_only_fields)
    # print "Fields added for target are:"
    # print list(added_target_fields)
    # print "Fields added for modeling are:"
    # print list(added_modeling_fields)
