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