# DTSC_300_First_Repository
First Repository for Big Data and Databases classes at Duquense

HW#2

Question 1: How do you fill in the missing dates from the grants data?
1. I filled missing dates by using the column 'Project Start', and if 
that did not have the value, I just used null. It reduced the number
of null entries from 264 to 3.

Question 2: PI_NAMEs contains multiple names. We can only connect individual people. Can you make it so that we can get "grantees"?

2. The simplest option in my opinion is to make a new dataframe with the grantees. This
dataframe now has a column for the application id and then the grantee, and has a row
for each of them

Code for these is available on grants.py

3. The dates for Articles are problematic. Can you fix them?
We fix PubDate and DateCompleted here
Before, it was getting tag text after pubDate, which was always \n.
We change this by going deeper into the XML structure, getting year, month, day
and then reformatting it as a string, storing this date. 

Code for these is available on articles.py


HW#3

Create the best possible classifier of sleep from acceleration and heart rate

Best Model Features I used (data from 8 people):

Wanted to normalize all features to a scale of each person, and then keep rolling averages
of heart rate and acceleration. I combined all acceleration into just a magnitude feature, 
as this would be more meaningful than having three seperate features.

'hr_norm'  -> Heart rate normalized for that person
'hr_norm_30m' -> mean normalized Heart rate for that person in the last 30 minutes
'acc_mag_norm' -> acceleration vector magnitude normalized for that person
'acc_mag_norm_30m' -> acceleration movement vector magnitude normalized for that person last 30 minutes
'acc_mag_norm_60m' -> acceleration movement vector magnitude normalized for that person last 60 minutes
'acc_mag_norm_480m' -> acceleration movemt vector magnitude normalized for that person last 8 hours
'hr_norm_60m' -> mean normalized Heart rate for that person in the last 60 minutes
'hr_norm_480m' -> mean normalized Heart rate for that person in the last 8 hours

Best Model Results I found:

Best threshold: 0.6724352673232223
Precision: 0.6461538461538462
Recall: 0.7
F1: 0.671999995008

              precision    recall  f1-score   support

       False       0.97      0.96      0.96       520
        True       0.65      0.70      0.67        60

    accuracy                           0.93       580
   macro avg       0.81      0.83      0.82       580
weighted avg       0.93      0.93      0.93       580

[[497  23]
 [ 18  42]]
ROC AUC: 0.9471153846153846
acc_mag_norm         0.271254
acc_mag_norm_30m     0.159674
hr_norm_480m         0.123686
acc_mag_norm_480m    0.120098
acc_mag_norm_60m     0.118934
hr_norm_60m          0.082632
hr_norm_30m          0.076491
hr_norm              0.047230

dtype: float64

Best threshold: 0.7163764470955606
Precision: 0.803921568627451
Recall: 0.6721311475409836
F1: 0.732142852182717
Fold 1 | AUC: 0.940 | Precision: 0.804 | Recall: 0.672 | F1: 0.732

Best threshold: 0.46816520847102866
Precision: 0.53
Recall: 0.8833333333333333
F1: 0.6624999953125
Fold 2 | AUC: 0.935 | Precision: 0.530 | Recall: 0.883 | F1: 0.662

Best threshold: 0.6373048624729275
Precision: 0.7678571428571429
Recall: 0.7166666666666667
F1: 0.741379305350773
Fold 3 | AUC: 0.946 | Precision: 0.768 | Recall: 0.717 | F1: 0.741

Best threshold: 0.596213979235793
Precision: 0.6949152542372882
Recall: 0.6833333333333333
F1: 0.6890756252524539
Fold 4 | AUC: 0.942 | Precision: 0.695 | Recall: 0.683 | F1: 0.689

Best threshold: 0.532619798572882
Precision: 0.676056338028169
Recall: 0.8
F1: 0.7328244225161705
Fold 5 | AUC: 0.940 | Precision: 0.676 | Recall: 0.800 | F1: 0.733

===== Cross-Validation Summary =====
AUC:       0.941 ± 0.004
Precision: 0.695 ± 0.095
Recall:    0.751 ± 0.080
F1 Score:  0.712 ± 0.031
