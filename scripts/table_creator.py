import sqlite3



def minimal_create_database():
    db_path = 'data/article_grant_db.sqlite'
    conn = sqlite3.connect(db_path)
    #cursor = conn.cursor() makes it possible to execute SQL commands on the database.
    

    
    conn.close()
    
    
if __name__ == "__main__":
    minimal_create_database()