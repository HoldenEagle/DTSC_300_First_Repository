# CamelCase
# snake_case -- this is what python programmers use

#Homework: add the same thing as this but for articles
from multiprocessing.dummy import connection

import pandas as pd
import sqlalchemy as SQLAlchemy
import re


class Grants():  # class names in python are camel case (e.g. GrantReader)
    def __init__(self, path: str | None = None):
        """Create and parse a Grants file

        Args:
            path (str): the location of the file on the disk
        """
        # What is self?
        # "Self is the specific instance of the object" - Computer Scientist 
        # Store shared variables in self
        self.path = path
        self.grantees = None
        self.df = self._parse(path)  # pi names in their own dataframe
        if path is None:
            self.df = self._from_db()
    
    def _from_db(self, db_path: str = 'data/article_grant_db.sqlite'):
        """Load the grants from a database"""
        engine = SQLAlchemy.create_engine(f'sqlite:///{db_path}')
        connection = engine.connect()
        df = pd.read_sql('SELECT * FROM grants', con=connection)
        connection.close()
        return df

    def _parse(self, path: str):
        """Parse a grants file"""
        df = pd.read_csv(path, compression='zip')
        mapper = {
            'APPLICATION_ID': 'application_id',  # _id means an id
            'BUDGET_START': 'start_at', #  _at means a date
            'ACTIVITY': 'grant_type',
            'TOTAL_COST': 'total_cost',
            'PI_NAMEs': 'pi_names',  # you will notice, homework references this
            'ORG_NAME': 'organization',
            'ORG_CITY': 'city',
            'ORG_STATE': 'state',
            'ORG_COUNTRY': 'country',
            'PROJECT_START': 'project_start',
        }
        # make column names lowercase
        # maybe combine for budget duration?
        df = df.rename(columns=mapper)[mapper.values()]
        #Handle missing dates: If start_at is missing, use project_start as a backup. 
        #If that's not there either, leave it as NaN.
        print(f"Number of missing start_at dates  edit: {df['start_at'].isna().sum()}")
        df['start_at'] = df['start_at'].fillna(df['project_start'])
        #Now we have to fix the multiple people in pi_names. Here we are going to make
        #another dataframe wiht the application id and the pi_name in each row
        
        #Eliminate rows with missing pi_names
        grantees = df.dropna(subset=['pi_names'])
        #split pi_names on ;
        grantees['pi_name'] = grantees['pi_names'].str.split(';')
        #puts it in its own rows
        grantees = grantees.explode('pi_name')
        #only columns we need
        grantees = grantees[['application_id', 'pi_name']]
        self.grantees = grantees
        return df
    
    def get(self):
        """Get parsed grants"""
        return self.df
    
    def get_grantees(self):
        return self.grantees 
    
    def to_db(self, db_path: str = 'data/article_grant_db.sqlite'):
        """Write the grants to a database"""
        engine = SQLAlchemy.create_engine(f'sqlite:///{db_path}')
        connection = engine.connect()

        # Only keep the columns you want
        df_to_insert = self.df[['application_id', 'start_at', 'grant_type', 'total_cost']]

        print(df_to_insert)

        df_to_insert.to_sql('grants', con=connection, if_exists='append', index=False)

        connection.close()
        
    
    #Got this code from CHATGPT, this cleans the pi_names by removing (contact) and middle initials,
    # and then puts them in the format "First Last"
    def grantees_to_db(self, db_path: str = 'data/article_grant_db.sqlite'):
        """Write the grantees to a database"""
    
        engine = SQLAlchemy.create_engine(f'sqlite:///{db_path}')
        connection = engine.connect()

        df_to_insert = self.grantees[['application_id', 'pi_name']].copy()

        def clean_name(name):
            if pd.isna(name):
                return name
        
            # Remove (contact)
            name = re.sub(r"\(contact\)", "", name, flags=re.IGNORECASE)
        
            # Split "LAST, FIRST ..."
            parts = name.split(",")

            if len(parts) == 2:
                last = parts[0].strip()
                first_part = parts[1].strip()

                # Remove middle initial (anything after first word)
                first = first_part.split()[0]

                name = f"{first} {last}"

            return name.title()

        df_to_insert["pi_name"] = df_to_insert["pi_name"].apply(clean_name)
        df_to_insert["pi_name"] = df_to_insert["pi_name"].str.lower()

        print(df_to_insert)

        df_to_insert.to_sql('grantees', con=connection, if_exists='append', index=False)

        connection.close()
    
    def _from_db_authors_bridge(self, db_path: str = 'data/article_grant_db.sqlite'):
        """Load the authors-grantees bridge table from a database"""
        engine = SQLAlchemy.create_engine(f'sqlite:///{db_path}')
        connection = engine.connect()
        df = pd.read_sql('SELECT * FROM author_grantee_bridge', con=connection)
        connection.close()
        return df


if __name__ == '__main__':
    # This is for debugging
    grants = Grants('C:\\Users\\holde\\DTSC_First_Repo\\DTSC_300_First_Repository\\data\\RePORTER_PRJ_C_FY2025.zip')
    grant_df = grants.get()
    print(grants.get_grantees().columns)
    #print(grant_df['start_at'].value_counts())
    #print(f"Number of missing start_at dates after edit: {grant_df['start_at'].isna().sum()}")
    print("-----------------------")
    
    #print(grant_df.head())
    print(grants.get_grantees().columns)
    print(grants.get_grantees().head())
    
    
    #grants = grants.get_grantees()
    #print(grants.head())
    #grants.to_db()
    #print(grants._from_db())  
    
    grants.grantees_to_db()
    
    print(grants._from_db_authors_bridge().head())
    
