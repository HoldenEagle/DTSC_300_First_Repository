"""
Read in Human Activity Recognition (HAR) data for classification of sleep
"""
import os
import pandas as pd
import numpy as np

class HAR():
    def __init__(self, path: str, n_people: int):
        """Take in a path, read in the data files

        Args:
            path (str): the BASE path of the directory
            n_people (int): the number of people's data to read
        """
        hrs = self._read_hr(path, n_people)
        mots = self._read_motion(path, n_people)
        lbls = self._read_labels(path, n_people)
        self.df = self._combine(hrs, mots, lbls)

    def _read_hr(self, path: str, n_people: int = 1):
        """Read the heartrate from the base path"""
        path = os.path.join(path, 'heart_rate')
        people = os.listdir(path)
        
        dfs = []
        count = 0
        for person in people:
            hrdf = pd.read_csv(os.path.join(path, person), names=['timestamp', 'hr'])
            hrdf['person'] = count
            dfs.append(hrdf)
            count = count + 1  # count += 1
            if count >= n_people:
                break

        return pd.concat(dfs)

    def _read_motion(self, path: str, n_people: int = 1):
        """Read the motion from the base path"""
        path = os.path.join(path, 'motion')
        people = os.listdir(path)
        
        dfs = []
        count = 0
        for person in people:
            motdf = pd.read_csv(os.path.join(path, person), delimiter=' ',names=['timestamp', 'acc_x', 'acc_y', 'acc_z'])
            motdf['person'] = count
            dfs.append(motdf)
            count = count + 1  # count += 1
            if count >= n_people:
                break

        return pd.concat(dfs)

    def _read_labels(self, path: str, n_people: int = 1):
        """Read the heartrate from the base path"""
        path = os.path.join(path, 'labels')
        people = os.listdir(path)
        
        dfs = []
        count = 0
        for person in people:
            print(person)
            labdf = pd.read_csv(os.path.join(path, person), names=['timestamp', 'label'], delimiter=' ')
            labdf['person'] = count
            dfs.append(labdf)
            count = count + 1  # count += 1
            if count >= n_people:
                break

        comb = pd.concat(dfs)
        comb['is_sleep'] = comb['label'] == 0
        return comb.loc[comb['label'] > -1].copy()  # This excludes missing data
    
    def _combine(self, hrs: pd.DataFrame, mots: pd.DataFrame, lbls: pd.DataFrame):
        """Combine three dataframes by interpolating to the highest sampling rate.
        The problem that we were having in class is that the timestamps were 
        collected separately. There are multiple ways to combine them, but this
        leaves the data closest to its raw state.
        """
        # In the case of multiple people, they may have been recorded 
        # with some overlap. We will fix that by using a recursive
        # function-- a function that calls itself.
        if len(pd.unique(hrs['person'])) > 1:
            people = pd.unique(hrs['person'])
            out = []
            for person in people:
                out.append(self._combine(hrs.loc[hrs['person'] == person],
                                         mots.loc[mots['person'] == person],
                                         lbls.loc[lbls['person'] == person]))
            return pd.concat(out)

        # Calculate median time interval between consecutive points for 
        # each dataframe
        # We will also exclude time points that aren't in all three dataframes
        min_interval = None
        min_interval_df = None
        last_start = None
        first_end = None
        dfs = {'hrs': hrs, 'mots': mots, 'lbls': lbls}
        sampling_intervals = {}
        for name, df in dfs.items():
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff().dropna()
            sampling_interval = time_diffs.median()

            if min_interval is None or sampling_interval < min_interval:
                min_interval = sampling_interval
                min_interval_df = name

            # Get the bounds of the recording time
            start_time = df_sorted['timestamp'].min()
            end_time = df_sorted['timestamp'].max()
            if last_start is None or start_time > last_start:
                last_start = start_time
            if first_end is None or end_time < first_end:
                first_end = end_time

        # Prepare each dataframe with timestamp as index
        for name in dfs:
            dfs[name] = dfs[name].copy()  # Copying prevents downstream errors
            dfs[name] = dfs[name].set_index('timestamp').sort_index()

        # Use the timestamps from the highest-frequency dataframe,
        # filtered to the time range where all three dataframes have data
        shared_index = dfs[min_interval_df].index
        shared_index = shared_index[(shared_index >= last_start) & (shared_index <= first_end)]

        # Interpolate each dataframe to the shared index
        # This is going to look VERY confusing. Let's walk through it.
        # 1. dfs['hrs'].reindex(
        # We're going to change the index of the dataframe.
        # 2. dfs['hrs'].index.union(shared_index))
        # This might look unnecessary, but we're making an index that 
        # includes BOTH the shared_index as well as the one from the
        # original dataframe. If we didn't and the timestamps didn't
        # perfectly match, we would end up with no data after using the
        # shared_index (because the timestamps specific to hrs wouldn't
        # exist within the timestamps for mots, which is the highest
        # frequency).
        # 3. .interpolate(method='index')
        # This computes the values in between by smoothly estimating them
        # .ffill fills them in forwards, which is appropriate for
        # categorical data
        # 4. .loc[shared_index]
        # Only keep the indices from mots.

        hrs_interp = dfs['hrs'].reindex(dfs['hrs'].index.union(shared_index)).interpolate(method='index').loc[shared_index]
        mots_interp = dfs['mots'].reindex(dfs['mots'].index.union(shared_index)).interpolate(method='index').loc[shared_index]
        lbls_interp = dfs['lbls'].reindex(dfs['lbls'].index.union(shared_index)).ffill().loc[shared_index]

        # Combine the dataframes
        combined = pd.concat([
            hrs_interp[['hr', 'person']],
            mots_interp[['acc_x', 'acc_y', 'acc_z']],
            lbls_interp[['is_sleep']]
        ], axis=1).dropna().reset_index()  # Remove NaNs and fix the index

        return combined


if __name__ == "__main__":
    #trasnform the raw data into features and labels for machine learning
    har = HAR(r'C:\Users\holde\DTSC_First_Repo\DTSC_300_First_Repository\data\motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0' , n_people=8)
    har = har.df
    print(len(har))
    #Sort by person and timestamp to ensure correct order for rolling features, and reset index
    har = har.sort_values(by=["person", "timestamp"]).reset_index(drop=True)
    WINDOW = 60  # 60 seconds = 1 minutes

    # Create time bins and aggregate features within each bin,
    # right now we compressing 60 1-second samples into 1 1-minute sample,
    # and we take the mean of the features within that minute.
    # later we create rolling features for 30m, 60m, and 480m (8 hours)
    har['time_bin'] = (har['timestamp'] // WINDOW).astype(int)
    har_5min = (
    har
    .groupby(['person', 'time_bin'], as_index=False)
    .agg({
        'timestamp': 'mean',   # representative time
        'hr': 'mean',
        'acc_x': 'mean',
        'acc_y': 'mean',
        'acc_z': 'mean',
        'is_sleep': 'mean'     # fraction of time asleep
    })
    )
    # is_sleep is an easy testing label, sleep fraction is a nice probability for ML
    har_5min = har_5min.rename(columns={'is_sleep': 'sleep_fraction'})
    har_5min['is_sleep_label'] = har_5min['sleep_fraction'] >= 0.5
    
    # Create normalized features by person
    har_5min['hr_norm'] = (
    har_5min['hr']
    - har_5min.groupby('person')['hr'].transform('mean')
    ) / har_5min.groupby('person')['hr'].transform('std')
    
    # Create magnitude of acceleration and normalize by person
    har_5min['acc_mag'] = np.sqrt(
    har_5min['acc_x']**2 +
    har_5min['acc_y']**2 +
    har_5min['acc_z']**2
    )
    har_5min['acc_mag_norm'] = (
    har_5min['acc_mag']
    - har_5min.groupby('person')['acc_mag'].transform('mean')
    ) / har_5min.groupby('person')['acc_mag'].transform('std')
    
    
    # Create rolling features for 30m, 60m, and 480m (8 hours), normalized by person
    har_5min['hr_norm_30m'] = (
    har_5min
    .groupby('person')['hr_norm']
    .rolling(30, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    )

    har_5min['acc_mag_norm_30m'] = (
    har_5min
    .groupby('person')['acc_mag_norm']
    .rolling(30, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    )
    
    har_5min['hr_norm_60m'] = (
    har_5min
    .groupby('person')['hr_norm']
    .rolling(60, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    )

    har_5min['acc_mag_norm_60m'] = (
    har_5min
    .groupby('person')['acc_mag_norm']
    .rolling(60, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    )
    
    har_5min['hr_norm_480m'] = (
    har_5min
    .groupby('person')['hr_norm']
    .rolling(480, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    )

    har_5min['acc_mag_norm_480m'] = (
    har_5min
    .groupby('person')['acc_mag_norm']
    .rolling(480, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    )
    
    #Create a new dataframe with just the features and labels we want for machine learning,
    # and save it as a csv file for easy loading in the ml_file.py script
    ml_df = har_5min[
    [
        'hr_norm',
        'hr_norm_30m',
        'acc_mag_norm',
        'acc_mag_norm_30m',
        'hr_norm_60m',
        'acc_mag_norm_60m',
        'hr_norm_480m',
        'acc_mag_norm_480m',
        'sleep_fraction',
        'is_sleep_label'
    ]
    ].dropna().reset_index(drop=True)
    
    ml_df.to_csv(r'C:\Users\holde\DTSC_First_Repo\DTSC_300_First_Repository\data\har_5min_features.csv', index=False)








    print(har_5min.head())