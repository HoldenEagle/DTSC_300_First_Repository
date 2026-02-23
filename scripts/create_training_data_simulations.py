#to run: python -m scripts.create_training_data_simulations
from readers.articles import Articles
import pandas as pd

class Training_Data_Simulation_Creation():
    def __init__(self):
        self.articles_df = Articles("data/pubmed26n1335.xml.gz").get_authors()
        self.training_df = self.create_training_df()
        
        
    def create_training_df(self):
        '''
        print(self.articles_df.columns)
        unique_names = self.articles_df[["ForeName", "LastName"]].dropna().drop_duplicates().sort_values(by=["LastName", "ForeName"])

        print(unique_names[100:150])
        total_rows = len(self.articles_df)
        '''

        total_rows = len(self.articles_df)
        n_matches = int(0.10 * total_rows)

        sampled = self.articles_df.sample(n=n_matches, random_state=42)

        training_rows = []

        for _, row in sampled.iterrows():
            row_dict = row.to_dict()

            pair = {
                "LastName_1": row_dict.get("LastName"),
                "ForeName_1": row_dict.get("ForeName"),
                "Initials_1": row_dict.get("Initials"),
                "Affiliation_1": row_dict.get("Affiliation"),

                "LastName_2": row_dict.get("LastName"),
                "ForeName_2": row_dict.get("ForeName"),
                "Initials_2": row_dict.get("Initials"),
                "Affiliation_2": row_dict.get("Affiliation"),

                "label": 1
            }

            training_rows.append(pair)
            
        n_non_matches = int(0.4 * total_rows)

        attempts = 0
        while len(training_rows) < n_matches + n_non_matches:
            row1 = self.articles_df.sample(1).iloc[0]
            row2 = self.articles_df.sample(1).iloc[0]

            if row1.equals(row2):
                continue

            match_count = 0
            for col in ["LastName", "ForeName", "Initials", "Affiliation"]:
                if row1[col] == row2[col]:
                    match_count += 1

            if match_count >= 3:
                continue

            pair = {
                "LastName_1": row1["LastName"],
                "ForeName_1": row1["ForeName"],
                "Initials_1": row1["Initials"],
                "Affiliation_1": row1["Affiliation"],

                "LastName_2": row2["LastName"],
                "ForeName_2": row2["ForeName"],
                "Initials_2": row2["Initials"],
                "Affiliation_2": row2["Affiliation"],

                "label": 0
            }

            training_rows.append(pair)
            
        n_partial = int(0.1 * total_rows)
        partial_sample = self.articles_df.sample(n=n_partial, random_state=101)

        for _, row in partial_sample.iterrows():
            row_dict = row.to_dict()
            pair = {
                "LastName_1": row_dict.get("LastName"),
                "ForeName_1": row_dict.get("ForeName"),
                "Initials_1": row_dict.get("Initials"),
                "Affiliation_1": row_dict.get("Affiliation"),

                "LastName_2": row_dict.get("LastName"),
                "ForeName_2": row_dict.get("ForeName"),
                "Initials_2": row_dict.get("Initials"),
                "Affiliation_2": row_dict.get("Affiliation"),

                "label": 1
            }

            import random
            cols_1 = ["LastName_2", "ForeName_2", "Initials_2", "Affiliation_2"]
            cols_2 = ["LastName_1", "ForeName_1", "Initials_1", "Affiliation_1"]
            which_side = random.choice(["left", "right", "both"])

            if which_side in ("left", "both"):
                col_to_null = random.choice(cols_1)
                pair[col_to_null] = None

            if which_side in ("right", "both"):
                col_to_null = random.choice(cols_2)
                pair[col_to_null] = None

            training_rows.append(pair)
            
        n_messup = int(0.3 * total_rows)
        partial_sample = self.articles_df.sample(n=n_messup, random_state=100)
        
        for _, row in partial_sample.iterrows():
            row_dict = row.to_dict()
            pair = {
                "LastName_1": row_dict.get("LastName"),
                "ForeName_1": row_dict.get("ForeName"),
                "Initials_1": row_dict.get("Initials"),
                "Affiliation_1": row_dict.get("Affiliation"),

                "LastName_2": row_dict.get("LastName"),
                "ForeName_2": row_dict.get("ForeName"),
                "Initials_2": row_dict.get("Initials"),
                "Affiliation_2": row_dict.get("Affiliation"),

                "label": 1
            }
            max_errors = 5
            cols_1 = ["LastName_1", "ForeName_1", "Initials_1", "Affiliation_1"]
            cols_2 = ["LastName_2", "ForeName_2", "Initials_2", "Affiliation_2"]
            all_cols = cols_1 + cols_2

            n_errors = random.randint(1, max_errors)
            for _ in range(n_errors):
                col = random.choice(all_cols)
                value = pair.get(col)
                if not value or not isinstance(value, str):
                    continue  # skip if None

                error_type = random.choice(["remove", "replace"])

            
                if error_type == "remove" and len(value) > 1:
                    idx = random.randint(0, len(value) - 1)
                    pair[col] = value[:idx] + value[idx+1:]
                elif error_type == "replace" and len(value) > 0:
                    idx = random.randint(0, len(value) - 1)
                    random_char = random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ  /'`")
                    pair[col] = value[:idx] + random_char + value[idx+1:]

            training_rows.append(pair)

        
        return pd.DataFrame(training_rows)
    
    def save_training_df(self):
        self.training_df.to_csv("data/training_data_phone_book.csv", index=False)
        
        



if __name__ == "__main__":
    tdsc = Training_Data_Simulation_Creation()
    tdsc.create_training_df()
    print(tdsc.articles_df)
    tdsc.save_training_df()
    