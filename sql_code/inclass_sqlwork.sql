/*
Select 100 rows from a table, including all columns.
Select only the forenmae or title column from a table.
Return the unique set of last names from a table.
Find all columns in which the forename is 'John'.
5. List employees alphabetically by last name.
6. Insert a new person or article into a table.
7. Then, change that person's forename.
8. Finally, delete that person.
9. Join together tables on forename, surname or on organization.
10. Only find identical matches
11. Find all possible matches for a single surname
12. If you don't have names in your db, you can either put them in or extract the first word of each title and treat it as a "name" SELECT substr(sentence, 1, instr(sentence, ' ') - 1) AS first_word FROM your_table;
How many unique surnames do you have? If not surnames, affiliations?

13. Find everyone who works at the same organization as a person named John (requires subqueries).

14. Find all people who work at an organization that appears more than once in the table (subqueries and cleverness)

15. If you complete this, what are the pandas equivalents for each?
*/




/* Question 1 */
/*Select 100 rows from a table, including all columns.*/

SELECT * FROM authors LIMIT 100;

/*Question 2 */
/* Select only the forenmae or title column from a table. */

SELECT ForeName FROM articles;

/*Question 3 */
/* Return the unique set of last names from a table. */

SELECT distinct(LastName) FROM articles;

/* Question 4 */
SELECT * FROM articles
WHERE ForeName = 'John';

/* Question 5 */
SELECT * FROM articles
WHERE LastName IS NOT NULL
order by LastName;

/* Question 6 */
INSERT INTO articles
VALUES(12345678, 'Test' , 'Test1' , 'TT' , 'Test University');

SELECT * FROM articles 
WHERE Affiliation = 'Test University';

/* Question 7 */
UPDATE articles
SET ForeName = 'Test2'
WHERE Affiliation = 'Test University';

SELECT * FROM articles 
WHERE Affiliation = 'Test University';

/*Question 8 */
DELETE FROM articles
WHERE id = 12345678;

SELECT * FROM articles 
WHERE id = 12345678;

/* Question 9 */
SELECT * FROM 
articles art
JOIN
authors auth
ON art.Affiliation = auth.affiliation;


/* Question 10 */
SELECT * FROM 
articles art
INNER JOIN
authors auth
ON art.Affiliation = auth.affiliation;

/* Question 11 */
WITH newtab as
  (SELECT * FROM authors2 LIMIT 1)
SELECT * FROM 
articles art
JOIN
newtab
ON art.LastName = newtab.LastName;


SELECT * FROM articles
WHERE LastName = 'Al Shobaili';

/*Question 12 */
SELECT COUNT(distinct(LastName))
FROM articles;


/* Question 13 */
WITH john_work AS
  (SELECT * FROM authors2 WHERE FirstName = 'John')
SELECT * FROM authors2
INNER JOIN
john_work
WHERE authors2.affiliation = john_work.affiliation
  AND authors2.FirstName <> 'John';


/*Question 14 */
WITH affiliation_count AS 
(
SELECT affiliation, COUNT(author_id) AS num_employees from authors2
GROUP BY affiliation
  HAVING affiliation IS NOT NULL
)
SELECT * FROM affiliation_count
WHERE num_employees > 1;