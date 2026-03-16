CREATE TABLE IF NOT EXISTS grants(
  id integer primary key autoincrement,
  application_id varchar(20) not null,
  start_at date,
  grant_type varchar(10),
  total_cost integer
);

CREATE TABLE authors (
    author_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pmid INTEGER,
    name TEXT,
    initials TEXT,
    affiliation TEXT
);


CREATE TABLE grantees (
    application_id INTEGER,
    pi_name TEXT
);