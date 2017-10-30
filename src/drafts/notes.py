# display unique events
print df_events['Event Name'].unique()

# count of users by country
df_users.groupby('Country').count().sort_values('Id', ascending=False)

X['Signed Up'].unique()
In[12]: X['Paid Subscription'].unique()
Out[12]: array([False,  True], dtype=bool)

In[13]: X['Practice'].unique()
Out[13]: array([False,  True], dtype=bool)

In[14]: X['Code Language'].unique()
Out[14]: array(['python', 'lua', 'javascript', 'coffeescript', 'java'], dtype=object)
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
max_user = df_users[df_users['Id'] == 268]  # from total_play_time

events_268 = df_events[df_events['User Id'] ==
                       268].sort_values(by="Created", ascending=False)

df_last_event.groupby('Id').count().sort_values('last_event_date')
df_last_event.groupby('last_action').count()
394, 98, and 471

df.isnull().any()
df['avg_num_days_per_level'].isnull().value_counts()
df['avg_L6_playtime_s'].isnull().value_counts()
c = l.groupby(['Levels Completed', 'last_campaign_started']).count()
c.sort_values('cat')
other = df_users[df_users['last_campaign_started'] == "other"]

df_users[['last_level_started'] == 'signs and portents']

# language counts in first six levels
find = df_levels[df_levels['Level'].isin(first_six_data)]
find['Code Language'].value_counts()

# country summary
countries = pd.DataFrame(df_users['Country'].value_counts())
countries['percent'] = countries['Country'] / 159817
countries[countries.index.isin(['united-states', 'united-kingdom',
                                'australia', 'canada', 'new-zealand', 'ireland', 'singapore'])].sum()
big = countries[countries['Country'] > 300]
# if set country cut off at 300, 4.7% of users will fall in to "other" country category
percent_captured = big.sum() / df_users.shape[0]

df_users['How old are you?'].value_counts()
df_levels['Level'].value_counts()


df_check = df_levels[df_levels['level_num'] == 60]['Level'].value_counts()
'''
Next steps:
Evaluate on test data and write up first ReadMe
Add % of first 6 played in each language
    filter on levels
    do unique on Code Language in those levels and iterate over languages
        represented to get a % variable for each language
Add number of logins in first 6
Add language cut off and add make an "other" category before dummifying
Clean up functions that deal with categorizing by level num or campaign

Tune model

Ask in scrum: if random forest can't overfit, why were results so different when ID number was accidentally included? Does that mean there is real signal there (based on time in month when user started)
'''
