import sqlite3



def minimal_create_database():
    db_path = 'data/article_grant_db.sqlite'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            LastName TEXT,
            ForeName TEXT,
            Initials TEXT,
            Affiliation TEXT
        )
    ''')

    conn.commit()
    conn.close()
    
    
if __name__ == "__main__":
    minimal_create_database()