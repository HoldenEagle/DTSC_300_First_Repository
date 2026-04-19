#first, read in the spam data called spam_email_dataset.csv in the data folder.
import pandas as pd
import numpy as np


#read in the data
data = pd.read_csv('../data/spam_email_dataset.csv')

#print out the column name and their data types
print(data.columns)
print(data.dtypes)


#lets make a feature called num_urgency_terms which counts the number of urgency terms in the email. We can use the str.count() function in pandas to count the number of times a word appears in a string.
urgency_terms = ['urgent', 'immediately', 'asap', 'important', 'attention', 'action required', 'deadline', 'hurry', 'now', 'today']
data['num_urgency_terms'] = data['email_text'].str.count('|'.join(urgency_terms))

#lets also make a feature called num_money_terms which counts the number of money related terms in the email. We can use the str.count() function in pandas to count the number of times a word appears in a string.
money_terms = ['money', 'cash', 'dollar', 'euro', 'pound', 'credit card', 'bank account', 'paypal', 'bitcoin', 'investment', 'loan', 'mortgage', 'debt', 'finance', 'financial', 'income', 'salary', 'paycheck']
data['num_money_terms'] = data['email_text'].str.count('|'.join(money_terms))

#lets make one for num_reward_terms which counts the number of reward related terms in the email. We can use the str.count() function in pandas to count the number of times a word appears in a string.
reward_terms = ['reward', 'prize', 'win', 'winner', 'congratulations', 'free', 'offer', 'deal', 'discount', 'coupon', 'gift', 'bonus', 'cashback']
data['num_reward_terms'] = data['email_text'].str.count('|'.join(reward_terms))


#lets find out which features are important. Lets compute the correlation between the features and the target variable, which is whether the email is spam or not. We can use the corr() function in pandas to compute the correlation matrix.
numeric_df = data.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
print(corr_matrix)
label_corr = corr_matrix['label'].sort_values(ascending=False)
print(label_corr)

'''
Found important features using correlation:
num_reward_terms ->           0.826316
num_urgency_terms ->         0.767838
num_money_terms ->         0.637371
contains_money_terms → 0.49 (we can drop this now that we have num_money_terms which is more informative)
contains_urgency_terms → 0.48 (drop this too)
num_words → 0.43 (eh we'll see if this is useful, but it is highly correlated with num_urgency_terms and num_money_terms, so we can drop this feature as well)
num_characters → 0.39 (lets drop this feature as it is highly correlated with num_words)
sender_reputation_score → -0.50
num_attachments → -0.33
has_attachment → -0.17 (but this can be found from num_attachments, so we can drop this feature)
'''

#lets use these and use an xgBoost classifier to predict whether an email is spam or not.
#We will use the following features above and use K-Fold cross validation to evaluate the model performance.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

#select the features and target variable
features = ['num_money_terms', 'num_urgency_terms', 'num_reward_terms', 'sender_reputation_score', 'num_attachments']
target = 'label'

X = data[features]
y = data[target]

#use K fold cross validation to evaluate the model performance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=5) #5 fold cross validation, didnt realize it does it for us lol
print("Cross validation scores: ", cv_scores)
print("Average cross validation score: ", np.mean(cv_scores))

#Now lets do the same with an XGBoost classifier and compare the results.
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100, random_state=42)
cv_scores_xgb = cross_val_score(xgb, X, y, cv=5)
print("Cross validation scores for XGBoost: ", cv_scores_xgb)
print("Average cross validation score for XGBoost: ", np.mean(cv_scores_xgb))


