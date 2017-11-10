# CodeCombat-capstone

## Summary
CodeCombat gamifies coding to teach users how to write code. The more levels users play, the better they will become at coding. In order to identify characteristics of players that affect how long they play and to identify specific members who are likely to churn early in order to try to intervene and affect their behavior, a collection of random forest models was created to predict churn at different points of the game. The model incorporates user demographics and features based on user behavior and experiences playing previous levels of the game. The resulting models exhibit predictive power for identifying churners. The model improves accuracy, precision, and F-1 scores over baseline. Random Forest feature importances were extracted and explored to make recommendations to CodeCombat on retaining players longer.

## Background
Coding is becoming a fundamental skill in the workplace,
but few elementary and middle school teachers are equipped to teach it.
CodeCombat is a company that fills this educational gap by gamifying coding to teach elementary
and middle school students to code. The game consists of hundreds of levels, organized into campaigns of different game themes. The more levels that students play, the more they practice coding and the better at coding they become.

CodeCombat is invested in retaining users so they can become better coders. Identifying characteristics of users and user behavior in early levels that result in relatively short-term game use could help the company work to develop strategies to maintain those users. Predicting churn at different game points for specific users would create an opportunity for the company to intervene with those users, for instance by offering them additional levels to play that are more appropriate for their skills, or reminding them about CodeCombat.  

## Data Insights
### Demographics
CodeCombat users come from many different countries, and are not limited to English-speaking countries. For users signing up in a single sample month, 52% were from outside the US. 31% of users signing up during this period were from countries in which English is not spoken natively.

Although the target audience of CodeCombat is young coders, the game has users of all ages. The majority of users reporting their age is in the 13 to 15 year old demographic, but 23% of users reporting their ages report 25 and older.

### User Behavior
Users generally play very early levels in the same order. After the first set of early levels, the pattern of play begins to diverge. Paying users have access to more levels, so they may play more levels in a campaign than non-paying users. Some users are offered practice levels at different points in the game which can also alter the pattern of user play.


## Modeling Methodology
The goal of the models was to predict user churn at different points throughout the game and then to extract information on drivers of churn at those points.

Data is from a set of users playing CodeCombat in a single month, with at least six months of additional behavior data.

Five models were created with different amounts of data available to each: Data from the first five levels played was used to predict if users would continue to play through level 10; data through level 10 was used to predict churn by level 15; data through level 30 was used to predict churn by level 60; and data through level 60 was used to predict churn by level 100.

Feature engineering included adding information on event-level actions the users took throughout the game, such as the number of hints the user clicked on, the number of times they started levels, etc.

Each of the five models was a Random Forest Classifier. Random Forest was chosen to give high model performance, but still provide insight into drivers of churn by extracting important features of each model. The parameters of each Random Forest were tuned via grid search.  

## Results
Compared to a baseline model predicting the majority class in each case, each of the Random Forest Models performs better than baseline with respect to accuracy, precision, and F-1 score.  

By using feature importance from the Random Forest models, it appears that later user churn is more actionable than earlier user churn. The top four important features for players churning between levels 61 and 100 were:
* Number of times started levels
* Number of recent practice levels
* Rate of bugs in recent user code
* Time to complete mid-range levels

## Recommendations
In order to retain members longer, CodeCombat should investigate suggesting practice levels more frequently and more aggressively. Analysis into the effects of changing practice levels played reveals that users who play more practice levels are less likely to churn. This could mean that playing practice levels encourages users to stay, or it could mean that the more dedicated users are more likely to be retained and more likely to play more practice levels.

CodeCombat should also investigate providing additional support for users experience bugs in their code. Analysis into this rate reveals that users with more bugs in later levels are more likely to churn. Additional resources or encouragement could influence this user behavior.
