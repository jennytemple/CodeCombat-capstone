import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#add something about number of hints uses/followed
#consider adding time spent playing last completed level
#add dummy variables for new fields if question unanswered
#add an indicator for churn (will take more work to define)

# first identify churners, then before churn occurred:
# last completed level
# time spent on last completed level
## last played level
## time spent on last played level
# number of logins for last played level
# number of hints used on last played level
## indicator if completed last level or not
# number of days between last logins

#Questions: should churners be broken into different groups? By either duration of account, number of logins, or number of levels played? For example, churners who leave between levels 10 and 49 may be different than those that churn between levels 50 and 149, or those who churn after 150. An early churner may be checking things out. A late churner may start finding the games too difficult or

def read_files(path):
    event_header = ['Created','User Id','Event Name','Level','raw_event1','raw_event2','raw_event3','raw_event4']

    df_users = pd.read_csv(path + 'users.csv')
    df_levels = pd.read_csv(path + 'levels.csv')
    df_events = pd.read_csv(path + 'events.csv', skiprows=1, names=event_header, error_bad_lines=False, warn_bad_lines=True)
    return (df_users, df_levels, df_events)

def user_filter(df_users, min_levels):
    df_users = df_users[df_users['Levels Completed'] >= min_levels ]
    return df_users

def add_total_play_time_per_user(df_users, df_levels):
    '''get total time spent playing per user and add to df'''
    total_play_time = df_levels[['Playtime (s)', 'User Id']].groupby('User Id').sum().sort_values(by ='Playtime (s)', ascending=False)
    #reset index
    total_play_time['Id'] = total_play_time.index

    #merge files
    df_users = pd.merge(df_users, total_play_time, how='left', on='Id')

    return df_users


def add_last_play_info(df_users, df_events):
    '''add information on last event - date, last action, and last level name'''

    #find latest event from event file
    df_last_event = df_events[['User Id', 'Created']].groupby(['User Id']).max()
    df_last_event['User Id'] = df_last_event.index #reset index

    #merge other data fields on to last_event
    df_last_event = pd.merge(df_last_event, df_events, how='left', on='User Id')
    #filter the resulting table for only the last events (but now with more information on the last events)
    df_last_event = df_last_event[df_last_event['Created_x'] == df_last_event['Created_y']]
    #drop unnecessaryfields
    df_last_event.drop(['Created_y', 'raw_event1', 'raw_event2', 'raw_event3', 'raw_event4'], axis=1, inplace=True)

    #dedup
    df_last_event = df_last_event.drop_duplicates()
    #rename fields
    df_last_event = df_last_event.rename(index=str, columns={'Created_x': 'last_event_date', 'User Id':'Id', 'Level':'last_level_name', 'Event Name': 'last_action'})

    #issue: if multiple event types occur at exactly the same time, both will be in the data set

    #join information to users data frame
    df_users = pd.merge(df_users, df_last_event, how='left', on='Id')
    return df_users

def add_active_days(df_users):
    '''add duration of membership in data set (creation date to last event date)'''
    #number of days between account creation and last action
    df_users['active_time_days'] = (pd.to_datetime(df_users['last_event_date']) - pd.to_datetime(df_users['Date Joined'])).dt.days
    #count the first day as active
    df_users['active_time_days']= df_users['active_time_days']+1
    return df_users


def add_activity_gap_info(df_users, data_pull_date):
    '''add number of days between last member activity and when data set was pulled'''
    df_users['data_through'] = pd.to_datetime(data_pull_date)
    df_users['activity_gap_days'] = (df_users['data_through']- pd.to_datetime(df_users['last_event_date'])).dt.days
    return df_users

def add_last_level_completion_info(df_users,df_levels):
    '''add field for if user last completed a level or was last mid-level'''
    #select last event
    df_last_level = df_levels[['User Id', 'Created']].groupby(['User Id']).max()
    #reset index
    df_last_level['User Id'] = df_last_level.index
    #merge rest of data on last event date
    df_last_level = pd.merge(df_last_level, df_levels, how='left', on='User Id')
    #select only the last event (but now with detail)
    df_last_level = df_last_level[df_last_level['Created_x'] == df_last_level['Created_y']]
    #add indicator for if the last level was completed
    df_last_level['was_completed'] = df_last_level['Date Completed'] > 0
    #drop unnecesary fields
    df_last_level.drop(['Created_y', 'Created_x', 'Date Completed'], axis=1, inplace=True)
    #rename columns
    df_last_level = df_last_level.rename(index=str, columns={'User Id': 'Id', 'Playtime (s)':'last_level_time_s'})

    #merge level data on to user data
    df_users = pd.merge(df_users, df_last_level, how='left', on='Id')
    return df_users

def add_avg_play_per_level(df_users):
    '''add the average play time per level for all levels played'''
    df_users['avg_play_time_per_level_s'] = df_users['Playtime (s)']/df_users['Levels Completed']
    return df_users

def add_avg_days_per_level(df_users):
    '''add the average levels/day for all levels played '''
    df_users['daygap'] = (pd.to_datetime(df_users['last_event_date']) - pd.to_datetime(df_users['Date Joined'])).dt.days + 1
    df_users['avg_num_days_per_level'] = df_users['daygap']/df_users['Levels Completed']
    return df_users

def add_last_level_started(df_users, df_levels):
    df_last_level_started = df_levels[['User Id', 'Created']].groupby(['User Id']).max()
    df_last_level_started['User Id'] = df_last_level_started.index
    df_last_level_started = pd.merge(df_last_level_started, df_levels, how='left', on='User Id')
    #select only the last event (but now with detail)
    df_last_level_started = df_last_level_started[df_last_level_started['Created_x'] == df_last_level_started['Created_y']]
    df_last_level_started['last_level_started'] = df_last_level_started['Level']
    df_last_level_started['Id'] = df_last_level_started['User Id']
    df_last_level_started = df_last_level_started[['Id','last_level_started']]
    df_last_level_started = df_last_level_started.drop_duplicates()
    df_users = pd.merge(df_users, df_last_level_started, how='left', on='Id')
    return df_users

def add_time_and_days_per(df_users, df_levels, level_group, name):
    '''add the average play time per level and average days per level for X levels (set by name)'''

    #filter level data according to the set
    df_levels_group = df_levels[df_levels['Level'].isin(level_group)]
    #get averge play time for these levels
    df_group_play = df_levels_group[['User Id', 'Playtime (s)']].groupby(['User Id']).sum()/len(level_group)
    #rename fields and merge onto users df
    df_group_play['Id'] = df_group_play.index

    #take max completion date
    df_levels_group_max = df_levels_group[['User Id', 'Date Completed']].groupby(['User Id']).max()
    df_levels_group_max['Id'] = df_levels_group_max.index

    #take min completion date
    df_levels_group_min = df_levels_group[['User Id', 'Created']].groupby(['User Id']).min()
    df_levels_group_min['Id'] = df_levels_group_min.index

    df_levels_group = pd.merge(df_levels_group_max, df_group_play, on='Id').merge(df_levels_group_min, on='Id')

    df_levels_group['avg_days_'+name] = (pd.to_datetime(df_levels_group['Date Completed']) - pd.to_datetime(df_levels_group['Created'])).dt.days

    df_levels_group.drop(['Date Completed','Created'], axis=1, inplace=True)
    df_levels_group = df_levels_group.rename(index=str, columns={'Playtime (s)':'avg_'+name+'_playtime_s'})

    df_users = pd.merge(df_users, df_levels_group, how='left', on='Id')

    return df_users

def add_number_special_activities(df_users, df_events, event_type, level_group, field_name):
    '''count the number of times per user a specific event type occured and add it so the users df '''
    df_events_group = df_events[df_events['Level'].isin(level_group)] #filter for levels
    df_events_group = df_events_group[df_events_group['Event Name']==event_type] #filter for event type
    df_events_group = df_events_group[['User Id','Event Name']].groupby(['User Id']).count() #count intances per user
    df_events_group['Id'] = df_events_group.index #replace index
    df_events_group = df_events_group.rename(index=str, columns={'Event Name': field_name})

    df_users = pd.merge(df_users, df_events_group, how='left', on='Id')
    df_users[field_name] = df_users[field_name].fillna(0)
    return df_users


if __name__=='__main__':
    sample_path = '../data/sample/'
    august_path = '../data/august/'
    march_path = '../data/march/'

    path = march_path
    df_users, df_levels, df_events = read_files(path)

    min_levels = 6
    df_users = user_filter(df_users, min_levels)

    df_users = add_total_play_time_per_user(df_users, df_levels) #add play time
    df_users = add_last_play_info(df_users, df_events)
    df_users = add_active_days(df_users)
    df_users = add_activity_gap_info(df_users,'2017-10-15')
    df_users = add_last_level_completion_info(df_users, df_levels)
    df_users = add_avg_play_per_level(df_users)
    df_users = add_avg_days_per_level(df_users)
    df_users = add_last_level_started(df_users, df_levels)

    #add average time to play and average days for groups of levels
    first_six_groups_business = ['dungeons-of-kithgard', 'gems-in-the-deep', 'enemy-mine', 'forgetful-gemsmith', 'shadow-guard', 'kounter-kithwise']
    first_six_data = ['dungeons-of-kithgard', 'gems-in-the-deep', 'shadow-guard', 'true-names', 'the-raised-sword', 'fire-dancing']
    #df_users = add_time_and_days_per(df_users, df_levels, first_six_group, "first_six")
    df_users = add_time_and_days_per(df_users, df_levels, ['dungeons-of-kithgard'], "L1")
    df_users = add_time_and_days_per(df_users, df_levels, ['gems-in-the-deep'], "L2")
    df_users = add_time_and_days_per(df_users, df_levels, ['shadow-guard'], "L3")
    df_users = add_time_and_days_per(df_users, df_levels, ['true-names'], "L4")
    df_users = add_time_and_days_per(df_users, df_levels, ['the-raised-sword'], "L5")
    df_users = add_time_and_days_per(df_users, df_levels, ['fire-dancing'], "L6")

    events_of_interest = ['Hint Used', 'Hints Clicked', 'Hints Next Clicked', 'Show problem alert']
    df_users = add_number_special_activities(df_users, df_events, 'Show problem alert', first_six_data, 'prob_alerts_first_six')
    df_users = add_number_special_activities(df_users, df_events, 'Hint Used', first_six_data, 'hints_used_first_six')
    df_users = add_number_special_activities(df_users, df_events, 'Hints Clicked', first_six_data, 'hints_clicked_first_six')

    #write out csv file for later use
    df_users.to_csv(path+'post_processed_users.csv')
