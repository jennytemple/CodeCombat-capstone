import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


event_header = ['Created','User Id','Event Name','Level','raw_event1','raw_event2','raw_event3','raw_event4']

df_users = pd.read_csv('../data/sample/users.csv')
df_levels = pd.read_csv('../data/sample/levels.csv')
df_events = pd.read_csv('../data/sample/events_orig.csv', skiprows=1, names=event_header, error_bad_lines=False, warn_bad_lines=True)

'''
df_users = pd.read_csv('../data/march/users.csv')
df_levels = pd.read_csv('../data/march/levels.csv')
df_events = pd.read_csv('../data/march/events.csv', skiprows=1, names=event_header, error_bad_lines=False, warn_bad_lines=True)
'''
'''
df_users = pd.read_csv('../data/august/users.csv')
df_levels = pd.read_csv('../data/august/levels.csv')
df_events = pd.read_csv('../data/august/events.csv', skiprows=1, names=event_header, error_bad_lines=False, warn_bad_lines=True)
'''

#display unique events
print df_events['Event Name'].unique()

#count of users by country
df_users.groupby('Country').count().sort_values('Id', ascending=False)

'''get total time spent playing per user and add to df'''
total_play_time = df_levels[['Playtime (s)', 'User Id']].groupby('User Id').sum().sort_values(by ='Playtime (s)', ascending=False)
total_play_time['Id'] = total_play_time.index

df_users = pd.merge(df_users, total_play_time, how='left', on='Id')


'''add information on last event - date, last action, and last level name'''
#find latest event from event file
df_last_event = df_events[['User Id', 'Created']].groupby(['User Id']).max()
#reset index
df_last_event['User Id'] = df_last_event.index
#merge other data fields on to last_event
df_last_event = pd.merge(df_last_event, df_events, how='left', on='User Id')
#filter the resulting table for only the last events (but now with more information on the last events)
df_last_event = df_last_event[df_last_event['Created_x'] == df_last_event['Created_y']]
#drop unnecessaryfields
df_last_event.drop(['Created_y', 'raw_event1', 'raw_event2', 'raw_event3', 'raw_event4'], axis=1, inplace=True)
#rename fields
df_last_event = df_last_event.rename(index=str, columns={'Created_x': 'last_event_date', 'User Id':'Id', 'Level':'last_level_name', 'Event Name': 'last_action'})
#join information to users data frame
df_users = pd.merge(df_users, df_last_event, how='left', on='Id')


'''add duration of membership in data set (creation date to last event date)'''
#number of days between account creation and last action
df_users['active_time_days'] = (pd.to_datetime(df_users['last_event_date']) - pd.to_datetime(df_users['Date Joined'])).dt.days
#count the first day as active
df_users['active_time_days']= df_users['active_time_days']+1

'''add number of days between last member activity and when data set was pulled'''
df_users['data_through'] = pd.to_datetime('2017-10-15')
df_users['activity_gap_days'] = (df_users['data_through']- pd.to_datetime(df_users['last_event_date'])).dt.days

'''add field for if user last completed a level or was last mid-level'''
#select last event
df_last_level = df_levels[['User Id', 'Created']].groupby(['User Id']).max()
#reset index
df_last_level['User Id'] = df_last_level.index
#merge rest of data on last event date
df_last_level = pd.merge(df_last_level, df_levels, how='left', on='User Id')
#select only the last event (but now with detail)
df_last_level = df_last_level[df_last_level['Created_x'] == df_last_level['Created_y']]
#add indicator for if the last level was completed (1) or not (0)
df_last_level['was_completed'] = df_last_level['Date Completed'] > 0
#drop unnecesary fields
df_last_level.drop(['Created_y', 'Created_x', 'Date Completed'], axis=1, inplace=True)
#rename columns
df_last_level = df_last_level.rename(index=str, columns={'User Id': 'Id', 'Playtime (s)':'last_level_time_s'})

'''merge level data on to user data '''
df_users = pd.merge(df_users, df_last_level, how='left', on='Id')
#consider changing column names to indicate last level played, and add data for last level completed


'''add the average play time per level for all levels played'''
df_users['avg_play_time_per_level_s'] = df_users['Playtime (s)']/df_users['Levels Completed']


'''add the average levels/day for all levels played '''
df_users['daygap'] = (pd.to_datetime(df_users['last_event_date']) - pd.to_datetime(df_users['Date Joined'])).dt.days + 1
df_users['avg_num_days_per_level'] = df_users['daygap']/df_users['Levels Completed']


'''add the average play time per level and average days per level for X levels (set by name)'''
#create a set of level names
first_six_levels = ['dungeons-of-kithgard', 'gems-in-the-deep', 'enemy-mine', 'forgetful-gemsmith', 'shadow-guard', 'kounter-kithwise']
#filter level data according to the set
df_first_six_levels = df_levels[df_levels['Level'].isin(first_six_levels)]
#get averge play time for these levels
df_first_six_play = df_first_six_levels[['User Id', 'Playtime (s)']].groupby(['User Id']).sum()/len(first_six_levels)
#rename fields and merge onto users df
df_first_six_play['Id'] = df_first_six_play.index

#take max completion date
df_first_six_levels = df_first_six_levels[['User Id', 'Date Completed']].groupby(['User Id']).max()
df_first_six_levels['Id'] = df_first_six_levels.index
df_first_six_levels = pd.merge(df_first_six_levels, df_first_six_play, on='Id')
df_first_six_levels = df_first_six_levels.rename(index=str, columns={'Playtime (s)':'avg_first_six_playtime_s', 'Date Completed':'date_completed_first_six'})

df_users = pd.merge(df_users, df_first_six_levels, how='left', on='Id')



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
'''
#other potentially useful things
df_users_over_10 = df_users[df_users['Levels Completed']>10]

df_reg_users_over_10 = df_users_over_10[df_users_over_10['Signed Up']==True]
df_reg_users = df_users[df_users['Signed Up']==True]
print df_reg_users['Levels Completed'].value_counts()
'''
'''
investigating a single users
'''
max_user = df_users[df_users['Id'] == 268] #from total_play_time

events_268 = df_events[df_events['User Id'] == 268].sort_values(by="Created", ascending=False)

def read_files(path):
    event_header = ['Created','User Id','Event Name','Level','raw_event1','raw_event2','raw_event3','raw_event4']

    df_users = pd.read_csv(path + 'users.csv')
    df_levels = pd.read_csv(path + 'levels.csv')
    df_events = pd.read_csv(path + 'events.csv', skiprows=1, names=event_header, error_bad_lines=False, warn_bad_lines=True)

if __name__=='__main__':
    sample_path = '../data/sample/''
    august_path = '../data/august/''
    march_path = '../data/march/''
    read_files(sample_path)
