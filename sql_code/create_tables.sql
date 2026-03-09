CREATE TABLE IF NOT EXISTS grants(
  id integer primary key autoincrement,
  application_id varchar(20) not null,
  start_at date,
  grant_type varchar(10),
  total_cost integer
);