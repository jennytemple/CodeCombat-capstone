# CodeCombat-capstone

## Summary
CodeCombat gamifies coding to teach users how to write code. The more levels users play, the better they will become at coding. In order to identify characteristics of players that affect how long they play and to identify specific members who are likely to churn early in order to try to intervene and affect their behavior, a random forest model was created to predict early, mid, and late churners. The model incorporates user demographics and features based on user behavior and experiences playing early levels of the game. The resulting model exhibits predictive power for identifying mid and late churners. The model improves the F-1 scores over baseline.  

## Background
Coding is becoming a fundamental skill in the workplace,
but few elementary and middle school teachers are equipped to teach it.
CodeCombat is a company that fills this educational gap by gamifying coding to teach elementary
and middle school students to code. The game consists of hundreds of levels, organized into campaigns of different game themes. The more levels that students play, the more they practice coding and the better at coding they become.

CodeCombat is invested in retaining users so they can become better coders. Identifying characteristics of users and user behavior in early levels that result in short-term game use could help the company work to develop strategies to maintain those users. Predicting churn for specific users would create an opportunity for the company to intervene with those users, for instance by offering them additional levels to play that are more appropriate for their skills, or reminding them about CodeCombat.  

## Data Insights
### Demographics
CodeCombat users come from many different countries, and are not limited to English-speaking countries. For users signing up in a single sample month, 52% were from outside the US. 31% of users signing up during this period were from countries in which English is not spoken natively.

Although the target audience of CodeCombat is young coders, the game has users of all ages. The majority of users reporting their age is in the 13 to 15 year old demographic, but 23% of users reporting their ages report 25 and older.

### User Behavior
Users generally play levels in the same order, but with many specific exceptions. Paying users have access to more levels, so they may play more levels in a campaign than non-paying users. Some users are offered practice levels at different points in the game which can also alter the pattern of user play.
Changing game

## Modeling Methodology
The goal of the model is to predict user churn from early user data and to extract the key features that indicate when users will churn.

Data is from a set of users playing CodeCombat in a single month, with at least six months of additional behavior data.

Data from the game levels that most users play as the first six levels was added to demographic data to model user behavior. The added features include, for instance, the average time a user takes playing the first six levels as well as the average number of special actions the user performs or sees during in these early levels. To avoid data leakage with the target, data is included for users only for the early levels the users complete. Any in-progress levels were not included.

Rather than modeling the absolute number of levels that users play, the goal of the model is to categorize users as early, mid, or late churners by binning the number of levels users play into these three categories. Another approach to model the campaign in which users finish playing was also investigated, but found to model behavior less well.

Eighty percent of the single-month data set was used for training the model while 20% was used to evaluate the model's performance.

Churn was modeled using a Multinomial Random Forest.

## Results
Compared to a baseline model using a single random feature, the random forest model performs significantly better when predicting the mid-churn or late-churn classes. The majority of users churn very early on in game play, so the baseline model, which was designed to contain no signal in the features, favors predicting most users as early churners.


## Next Steps
* Add additional features:
  * Percent of early levels played in each programming language
  * Average number of logins a user has for the early levels


* Consider increasing the number of early level data used to model later behavior

* Balance classes

* Tune parameters of the random forest model, including the divisions used to define early, mid, and late churners.

* Clearly segregate data into train/test splits and use k-fold cross validation for model tuning
