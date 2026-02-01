# CamelCase
# snake_case -- this is what python programmers use
import pandas as pd


class Grants():  # class names in python are camel case (e.g. GrantReader)
    def __init__(self, path: str):
        """Create and parse a Grants file

        Args:
            path (str): the location of the file on the disk
        """
        # What is self?
        # "Self is the specific instance of the object" - Computer Scientist 
        # Store shared variables in self
        self.path = path
        self.df = self._parse(path)
        self.grantees = None  # pi names in their own dataframe

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


if __name__ == '__main__':
    # This is for debugging
    grants = Grants('C:\\Users\\holde\\DTSC_First_Repo\\DTSC_300_First_Repository\\data\\RePORTER_PRJ_C_FY2025.zip')
    grant_df = grants.get()
    #print(grant_df['start_at'].value_counts())
    print(f"Number of missing start_at dates after edit: {grant_df['start_at'].isna().sum()}")
    print("-----------------------")
    
    #Answer to number 1, getting indivual grantees.
    grantees = grants.get_grantees()
    print(grantees.head(10))
    
