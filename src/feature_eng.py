import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_files(path):
    event_header = ['Created', 'User Id', 'Event Name', 'Level',
                    'raw_event1', 'raw_event2', 'raw_event3', 'raw_event4']

    df_users = pd.read_csv(path + 'users.csv')
    if 'Unnamed: 0' in df_users.columns:
        df_users.drop('Unnamed: 0', axis=1, inplace=True)
    df_levels = pd.read_csv(path + 'levels.csv')
    df_events = pd.read_csv(path + 'events.csv', skiprows=1,
                            names=event_header, error_bad_lines=False, warn_bad_lines=True)
    return (df_users, df_levels, df_events)


def cleanup_events(df_events):
    df_events.drop(['raw_event1', 'raw_event2', 'raw_event3',
                    'raw_event4'], axis=1, inplace=True)
    df_events['Created'] = pd.to_datetime(df_events['Created'])
    return df_events


def cleanup_levels(df_levels):
    df_levels['Created'] = pd.to_datetime(df_levels['Created'])
    return df_levels


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
    df_users = pd.merge(df_users, total_play_time, how='left', on='Id')

    return df_users


def add_last_play_info(df_users, df_events):
    '''add information on last event - date, last action, and last level name'''

    # find latest event from event file
    df_last_event = df_events[['User Id', 'Created']
                              ].groupby(['User Id']).max()
    df_last_event['User Id'] = df_last_event.index  # reset index

    # merge other data fields on to last_event
    df_last_event = pd.merge(df_last_event, df_events,
                             how='left', on='User Id')
    # filter the resulting table for only the last events (but now with more information on the last events)
    df_last_event = df_last_event[df_last_event['Created_x']
                                  == df_last_event['Created_y']]
    # drop unnecessaryfields
    df_last_event.drop(['Created_y'], axis=1, inplace=True)

    # dedup
    df_last_event = df_last_event.drop_duplicates()
    # rename fields
    df_last_event = df_last_event.rename(index=str, columns={
                                         'Created_x': 'last_event_date', 'User Id': 'Id', 'Level': 'last_level_name', 'Event Name': 'last_action'})

    # issue: if multiple event types occur at exactly the same time, both will be in the data set

    # join information to users data frame
    df_users = pd.merge(df_users, df_last_event, how='left', on='Id')
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
    df_last_level = pd.merge(df_last_level, df_levels,
                             how='left', on='User Id')
    # select only the last event (but now with detail)
    df_last_level = df_last_level[df_last_level['Created_x']
                                  == df_last_level['Created_y']]
    # add indicator for if the last level was completed
    df_last_level['last_level_was_completed'] = df_last_level['Date Completed'] > 0
    # drop unnecesary fields
    df_last_level.drop(
        ['Created_y', 'Created_x', 'Date Completed', 'Code Language'], axis=1, inplace=True)
    # rename columns
    df_last_level = df_last_level.rename(
        index=str, columns={'User Id': 'Id', 'Playtime (s)': 'last_level_time_s', 'Practice': 'last_level_was_practice', 'Level': 'last_level_played'})

    # merge level data on to user data
    df_users = pd.merge(df_users, df_last_level, how='left', on='Id')
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


def add_last_level_started(df_users, df_levels):
    df_levels = df_levels.dropna(axis=0, how='any')
    df_last_level_started = df_levels[['User Id', 'Created']].groupby([
                                                                      'User Id']).max()
    df_last_level_started['User Id'] = df_last_level_started.index
    df_last_level_started = pd.merge(
        df_last_level_started, df_levels, how='left', on='User Id')
    # select only the last event (but now with detail)
    df_last_level_started = df_last_level_started[df_last_level_started['Created_x']
                                                  == df_last_level_started['Created_y']]
    df_last_level_started['last_level_started'] = df_last_level_started['Level']
    df_last_level_started['Id'] = df_last_level_started['User Id']
    df_last_level_started = df_last_level_started[['Id', 'last_level_started']]
    df_last_level_started = df_last_level_started.drop_duplicates()
    df_users = pd.merge(df_users, df_last_level_started, how='left', on='Id')
    return df_users


def add_last_campaign_started(df_users):
    campaigns = pd.read_csv('../../data/campaign_list.csv')
    campaigns = campaigns.rename(index=str, columns={
                                 'Campaign': 'last_campaign_started', 'Level': 'last_level_started'})
    df_users = pd.merge(df_users, campaigns, how='left',
                        on='last_level_started')
    df_users['last_campaign_started'] = df_users['last_campaign_started'].fillna(
        'other')
    return df_users


def add_group_completion_date(df_users, df_events, level_group, name):
    # filter events for level completions of groups
    df_events_group = df_events[df_events['Level'].isin(level_group)]
    df_events_group = df_events_group[df_events_group['Event Name']
                                      == 'Saw Victory']

    # take max date and add to dataframe
    df_events_group_max = df_events_group[[
        'User Id', 'Created']].groupby('User Id').max()
    df_events_group_max['Id'] = df_events_group_max.index
    df_events_group_max['date_completed_' +
                        name] = pd.to_datetime(df_events_group_max['Created'])
    df_events_group_max.drop(['Created'], axis=1, inplace=True)

    # merge on user data
    df_users = pd.merge(df_users, df_events_group_max, how='left', on='Id')
    return df_users


def add_group_completion_num(df_users, df_events, level_group, name):
    # filter events for level completions of groups
    df_events_group = df_events[df_events['Level'].isin(level_group)]
    df_events_group = df_events_group[df_events_group['Event Name']
                                      == 'Saw Victory']
    # take max date and add to dataframe
    df_events_group_count = df_events_group[[
        'User Id', 'Event Name']].groupby('User Id').count()
    df_events_group_count['Id'] = df_events_group_count.index
    df_events_group_count = df_events_group_count.rename(
        index=str, columns={'Event Name': 'num_levels_completed_in_' + name})

    # merge on user data
    df_users = pd.merge(df_users, df_events_group_count, how='left', on='Id')
    return df_users


def add_number_special_activities(df_users, df_events, event_type, date_field, ref_field, new_field_name, use_events=True):
    '''count the number of times per user a specific event type occured within a time frame defined earlier and add it to the users df '''
    '''primarily built for event dataframe events, but also handles practice field in the level data frame'''
    '''Note: most special events are not tied to a particular level, so time frame has to be used instead of level '''
    if use_events:
        df_events_special = df_events[df_events['Event Name'] == event_type]
    else:
        df_events_special = df_events[df_events['Practice'] == True]

    df_events_special = df_events_special.rename(
        index=str, columns={'User Id': 'Id', 'Created': 'event_date'})
    df_users_max_dates = df_users[['Id', date_field]]

    df_events_special = pd.merge(
        df_events_special, df_users_max_dates, how='left', on='Id')
    df_events_special = df_events_special[df_events_special['event_date']
                                          <= df_events_special[date_field]]
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
    df_users = pd.merge(df_users, event_count, how='left', on='Id')
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
    countries = pd.DataFrame(df_users['Country'].value_counts())
    countries_included = countries[countries['Country'] > n]
    include = list(countries_included.index)
    df_users['Country'] = df_users.apply(
        lambda row: row['Country'] if row['Country'] in include else "other_country", axis=1)

    return df_users


def add_coding_language(df_users, df_levels, level_group, num_levels_col):
    '''Note: final column added may be greater than 1.0. Some levels are "completed" without seeing victory. Look into "the-raised-sword" to see detail'''
    df_user_num_levels = df_users[['Id', num_levels_col]]

    # include only completed levels
    df_levels = df_levels[df_levels['Date Completed'].notnull()]
    df_levels = df_levels[df_levels['Level'].isin(
        level_group)]  # include only levels in list

    languages = list(df_levels['Code Language'].unique())
    for l in languages:
        df_levels_l = df_levels[df_levels['Code Language'] == l]  # filter
        df_levels_l = df_levels_l[['User Id', 'Code Language']].groupby(
            'User Id').count()
        df_levels_l['Id'] = df_levels_l.index
        df_levels_l = pd.merge(
            df_levels_l, df_user_num_levels, how='left', on='Id')
        df_levels_l[l + '_in_first_six'] = df_levels_l['Code Language'] / \
            df_levels_l[num_levels_col]
        df_levels_l.drop(['Code Language', num_levels_col],
                         axis=1, inplace=True)
        df_users = pd.merge(df_users, df_levels_l, how='left', on='Id')
        df_users[l + '_in_first_six'] = df_users[l + '_in_first_six'].fillna(0)

    return df_users


def add_number_logins(df_users, df_events, max_date_col, time_threshold_hours=1.0):
    '''for each user, and each event, add time between events, filter on max date, and count the number of events over a certain time '''
    # filter dates
    df_users_max_dates = df_users[['Id', max_date_col]]
    df_users_max_dates = df_users_max_dates.rename(
        index=str, columns={'Id': 'User Id'})
    df_events = pd.merge(df_events, df_users_max_dates,
                         how='left', on='User Id')
    df_events = df_events[df_events['Created'] <= df_events[max_date_col]]

    # get time between events in hours
    df_events = df_events.sort_values(
        'Created')  # data set appears to be sorted by time already, but repeat in case that assumption is incorrect
    df_events['shifted'] = df_events.groupby('User Id')['Created'].shift(1)
    df_events['lag'] = df_events['shifted'] - df_events['Created']
    df_events['lag_days'] = (df_events['Created'] -
                             df_events['Created'].shift()).dt.days
    df_events['lag_secs'] = (df_events['Created'] -
                             df_events['Created'].shift()).dt.seconds
    df_events['lag_hours'] = (df_events['lag_days'] * 24.0) + \
        (df_events['lag_secs'] / 60 / 60)

    # get number of events exceeding time threshold
    df_events = df_events[df_events['lag_hours'] >= time_threshold_hours]
    df_logins = df_events[['User Id', 'lag_hours']].groupby('User Id').count()
    df_logins['Id'] = df_logins.index

    df_users = pd.merge(df_users, df_logins, how='left', on='Id')
    return df_users


if __name__ == '__main__':
    sample_path = '../../data/sample/'
    august_path = '../../data/august/'
    march_path = '../../data/march/'
    tiny_sample_path = '../../data/tiny_sample/'

    path = march_path
    df_users, df_levels, df_events = read_files(path)

    # clean up and filter user df if necessary
    df_events = cleanup_events(df_events)
    df_levels = cleanup_levels(df_levels)
    df_users = cleanup_users(df_users)
    min_level_reached = 1  # 7
    df_users = user_filter(df_users, min_level_reached)
    orig_fields = set(df_users.columns)

    # add fields to user df for eda (not to be used in model)
    df_users = add_total_play_time_per_user(
        df_users, df_levels)  # add play time
    df_users = add_last_play_info(df_users, df_events)
    df_users = add_active_days(df_users)
    df_users = add_activity_gap_info(df_users, '2017-10-15')
    df_users = add_last_level_completion_info(df_users, df_levels)
    df_users = add_avg_play_per_level(df_users)
    df_users = add_avg_days_per_level(df_users)
    added_eda_only_fields = set(df_users.columns) - orig_fields

    # add fields to user df for target modeling and eda
    df_users = add_last_level_started(df_users, df_levels)
    df_users = add_last_campaign_started(df_users)
    added_target_fields = set(df_users.columns) - \
        orig_fields - added_eda_only_fields

    # add fields to user df for features modeling and eda
    df_users = fill_out_age(df_users)
    # df_users = add_country_group(df_users)
    df_users = add_english_speaking(df_users)
    df_users = aggregate_small_pop_countries(df_users, n=300)

    # add average time to play and average days for teh first six levels
    first_six_data = ['dungeons-of-kithgard', 'gems-in-the-deep',
                      'shadow-guard', 'true-names', 'the-raised-sword', 'fire-dancing']
    df_users = add_group_completion_date(
        df_users, df_events, first_six_data, "first_six")
    df_users = add_group_completion_num(
        df_users, df_events, first_six_data, "first_six")

    df_users = add_number_special_activities(
        df_users, df_events, 'Hint Used', 'date_completed_first_six', 'num_levels_completed_in_first_six', 'rate_hint_used_first_six')
    df_users = add_number_special_activities(
        df_users, df_events, 'Hints Clicked', 'date_completed_first_six', 'num_levels_completed_in_first_six', 'rate_hints_clicked_first_six')
    df_users = add_number_special_activities(
        df_users, df_events, 'Hints Next Clicked', 'date_completed_first_six', 'num_levels_completed_in_first_six', 'rate_hints_next_clicked_first_six')
    df_users = add_number_special_activities(
        df_users, df_events, 'Started Level', 'date_completed_first_six', 'num_levels_completed_in_first_six', 'rate_started_level_first_six')
    df_users = add_number_special_activities(
        df_users, df_events, 'Show problem alert', 'date_completed_first_six', 'num_levels_completed_in_first_six', 'rate_show_problem_alerts_first_six')
    df_users = add_number_special_activities(
        df_users, df_levels, 'Practice', 'date_completed_first_six', 'num_levels_completed_in_first_six', 'rate_practice_levels_first_six', use_events=False)
    df_users = add_coding_language(
        df_users, df_levels, first_six_data, 'num_levels_completed_in_first_six')
    df_users = add_number_logins(
        df_users, df_events, 'date_completed_first_six', time_threshold_hours=1.0)

    added_modeling_fields = set(df_users.columns) - orig_fields - \
        added_eda_only_fields - added_target_fields

    # for testing. Maybe for EDA...
    # all_levels = list(df_levels['Level'].unique())
    # df_users = add_group_completion_date(
    #     df_users, df_events, all_levels, "all")
    # df_users = add_group_completion_num(
    #     df_users, df_events, all_levels, "all")
    # df_users = add_number_special_activities(
    #     df_users, df_levels, 'Practice', 'date_completed_all', 'num_levels_completed_in_all', 'practice_levels_all', use_events=False)

    # write out csv file for later use
    df_users.to_csv(path + 'post_processed_users.csv')

    print "Orig fields are:"
    print list(orig_fields)
    print "Fields added for EDA are:"
    print list(added_eda_only_fields)
    print "Fields added for target are:"
    print list(added_target_fields)
    print "Fields added for modeling are:"
    print list(added_modeling_fields)
