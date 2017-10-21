import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# add something about number of hints uses/followed
# consider adding time spent playing last completed level
# add dummy variables for new fields if question unanswered
# add an indicator for churn (will take more work to define)

# first identify churners, then before churn occurred:
# last completed level
# time spent on last completed level
# last played level
# time spent on last played level
# number of logins for last played level
# number of hints used on last played level
# indicator if completed last level or not
# number of days between last logins

# Questions: should churners be broken into different groups? By either duration of account, number of logins, or number of levels played? For example, churners who leave between levels 10 and 49 may be different than those that churn between levels 50 and 149, or those who churn after 150. An early churner may be checking things out. A late churner may start finding the games too difficult or

def read_files(path):
    event_header = ['Created', 'User Id', 'Event Name', 'Level',
                    'raw_event1', 'raw_event2', 'raw_event3', 'raw_event4']

    df_users = pd.read_csv(path + 'users.csv')
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
        ['Created_y', 'Created_x', 'Date Completed', 'Unnamed: 0', 'Code Language'], axis=1, inplace=True)
    # rename columns
    df_last_level = df_last_level.rename(
        index=str, columns={'User Id': 'Id', 'Playtime (s)': 'last_level_time_s', 'Practice': 'last_level_was_practice', 'Level': 'last_level_played'})

    # merge level data on to user data
    df_users = pd.merge(df_users, df_last_level, how='left', on='Id')
    return df_users


def add_avg_play_per_level(df_users):
    '''add the average play time per level for all levels played'''
    df_users['avg_play_time_per_level_s'] = df_users['Playtime (s)'] / \
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


'''
deprecated
'''
# def add_time_and_days_per(df_users, df_levels, level_group, name):
#     '''add the average play time per level and average days per level for X levels (set by name)'''
#
#     # filter level data according to the set
#     df_levels_group = df_levels[df_levels['Level'].isin(level_group)]
#
#     # count number of levels started (each level has one entry in table)
#     counts = df_levels_group.groupby('User Id').count()
#     counts = counts['Level'].to_frame()
#     counts['Id'] = counts.index
#     # get averge play time for these levels
#     df_group_play = df_levels_group[['User Id', 'Playtime (s)']].groupby(
#         ['User Id']).sum()
#     df_group_play['Id'] = df_group_play.index
#     df_group_play = pd.merge(df_group_play, counts, how='left', on='Id')
#     df_group_play['avg_' + name +
#                   '_playtime_s'] = df_group_play['Playtime (s)'] / df_group_play['Level']
#
#     # take max completion date
#     # note: since not all levels are completed, need to consider level creation date as well
#     df_levels_max_cand1 = df_levels_group[['User Id']]
#     df_levels_max_cand2 = df_levels_group[['User Id', 'Created']]
#     df_levels_max_cand2 = df_levels_max_cand2.rename(
#         index=str, columns={'Created': 'Date Completed'})
#     combine = [df_levels_max_cand1, df_levels_max_cand2]
#
#     df_levels_group_max = pd.concat(combine)
#     df_levels_group_max = df_levels_group_max[['User Id', 'Date Completed']].groupby([
#         'User Id']).max()
#     df_levels_group_max['Id'] = df_levels_group_max.index
#
#     # take min completion date
#     df_levels_group_min = df_levels_group[[
#         'User Id', 'Created']].groupby(['User Id']).min()
#     df_levels_group_min['Id'] = df_levels_group_min.index
#
#     # merge min and max onto main dataframe
#     df_levels_group = pd.merge(df_levels_group_max, df_group_play, on='Id').merge(
#         df_levels_group_min, on='Id')
#
#     df_levels_group['avg_days_' + name] = (pd.to_datetime(
#         df_levels_group['Date Completed']) - pd.to_datetime(df_levels_group['Created'])).dt.days + 1
#
#     df_levels_group.drop(['Playtime (s)', 'Created'], axis=1, inplace=True)
#     df_levels_group = df_levels_group.rename(index=str, columns={
#                                              'Date Completed': 'date_completed_' + name, 'Level': 'num_levels_completed_in_' + name})
#
#     df_users = pd.merge(df_users, df_levels_group, how='left', on='Id')
#
#     return df_users
'''
deprecated
'''
# def add_number_special_activities(df_users, df_events, event_type, level_group, field_name):
#     '''count the number of times per user a specific event type occured attached to specific levels names, and add it so the users df '''
#     df_events_group = df_events[df_events['Level'].isin(
#         level_group)]  # filter for levels
#     # filter for event type
#     df_events_group = df_events_group[df_events_group['Event Name'] == event_type]
#     df_events_group = df_events_group[['User Id', 'Event Name']].groupby(
#         ['User Id']).count()  # count intances per user
#     df_events_group['Id'] = df_events_group.index  # replace index
#     df_events_group = df_events_group.rename(
#         index=str, columns={'Event Name': field_name})
#
#     df_users = pd.merge(df_users, df_events_group, how='left', on='Id')
#     df_users[field_name] = df_users[field_name].fillna(0)
#     return df_users


def add_number_special_activities(df_users, df_events, event_type, date_field, new_field_name):
    '''count the number of times per user a specific event type occured within a time frame defined earlier and add it to the users df '''
    df_events_special = df_events[df_events['Event Name'] == event_type]
    df_events_special = df_events_special.rename(
        index=str, columns={'User Id': 'Id', 'Created': 'event_date'})
    df_users_max_dates = df_users[['Id', date_field]]

    df_events_special = pd.merge(
        df_events_special, df_users_max_dates, how='left', on='Id')
    df_events_special = df_events_special[df_events_special['event_date']
                                          <= df_events_special[date_field]]
    event_count = df_events_special.groupby('Id').count()
    event_count = event_count.rename(
        index=str, columns={'Event Name': new_field_name})
    event_count = event_count[[new_field_name]]
    event_count['Id'] = event_count.index.astype(int)

    # merge and fill Nans with zeros
    df_users = pd.merge(df_users, event_count, how='left', on='Id')
    df_users[new_field_name] = df_users[new_field_name].fillna(0)
    df_users[new_field_name] = df_users[new_field_name] / \
        df_users['num_levels_completed_in_first_six']

    return df_users


def add_num_startlevels(df_users, df_events, date_field, event_type='Started Level'):
    df_users = add_number_special_activities(
        df_users, df_events, event_type, date_field, 'temp_count')
    df_users['starts_per_played_first_six'] = df_users['temp_count'] / \
        df_users['num_levels_completed_in_first_six']
    df_users.drop(['temp_count'], axis=1, inplace=True)

    '''
    Investigate problem: some users with fewer starts than levels played? df_users[df_users['starts_per_played_first_six']<1.0]
    '''
    return df_users


def fill_out_age(df_users):
    df_users['How old are you?'] = df_users['How old are you?'].fillna('other')
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


def add_coding_language(df_users, df_levels):
    '''IN PROGRESS'''
    language_counts = df_levels.groupby(
        ['Code Language', 'User Id']).size().reset_index(name='counter')
    pass


if __name__ == '__main__':
    sample_path = '../../data/sample/'
    august_path = '../../data/august/'
    march_path = '../../data/march/'
    tiny_sample_path = '../../data/tiny_sample/'

    path = tiny_sample_path
    df_users, df_levels, df_events = read_files(path)
    print df_users.columns

    # clean up and filter user df if necessary
    df_events = cleanup_events(df_events)
    df_users = cleanup_users(df_users)
    min_level_reached = 1  # 7
    df_users = user_filter(df_users, min_level_reached)

    # add fields to user df for eda (not to be used in model)
    df_users = add_total_play_time_per_user(
        df_users, df_levels)  # add play time
    df_users = add_last_play_info(df_users, df_events)
    df_users = add_active_days(df_users)
    df_users = add_activity_gap_info(df_users, '2017-10-15')
    print df_users.columns
    df_users = add_last_level_completion_info(df_users, df_levels)
    df_users = add_avg_play_per_level(df_users)

    # add fields to user df for target modeling and eda
    df_users = add_avg_days_per_level(df_users)
    df_users = add_last_level_started(df_users, df_levels)
    df_users = add_last_campaign_started(df_users)

    # add fields to user df for features modeling and eda
    df_users = fill_out_age(df_users)
    # df_users = add_country_group(df_users)
    df_users = add_english_speaking(df_users)

    # add average time to play and average days for teh first six levels
    first_six_data = ['dungeons-of-kithgard', 'gems-in-the-deep',
                      'shadow-guard', 'true-names', 'the-raised-sword', 'fire-dancing']
    df_users = add_group_completion_date(
        df_users, df_events, first_six_data, "first_six")
    df_users = add_group_completion_num(
        df_users, df_events, first_six_data, "first_six")

    df_users = add_number_special_activities(
        df_users, df_events, 'Hint Used', 'date_completed_first_six', 'hint_used_first_six')
    df_users = add_number_special_activities(
        df_users, df_events, 'Hints Clicked', 'date_completed_first_six', 'hints_clicked_first_six')
    df_users = add_number_special_activities(
        df_users, df_events, 'Hints Next Clicked', 'date_completed_first_six', 'hints_next_clicked_first_six')
    df_users = add_number_special_activities(
        df_users, df_events, 'Show problem alert', 'date_completed_first_six', 'show_problem_alerts_first_six')
    df_users = add_num_startlevels(
        df_users, df_events, 'date_completed_first_six')

    # write out csv file for later use
    df_users.to_csv(path + 'post_processed_users.csv')

    print "cool"
