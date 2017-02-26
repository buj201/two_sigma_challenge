import pandas as pd
import numpy as np
import os

def format_monthly_data(df):
    """ Formats monthly trip data by:
            - Constructing datetime features
            - Dropping records with 0 lat/longs
            - Filling missing ages using the stations monthly mean age
            - Makes user type dummies

        Returns:
            - df: Formatted pandas DataFrame

    """
    #Make datetime features
    df['Day of Week'] = df['Start Time'].dt.dayofweek
    df['Hour'] = df['Start Time'].dt.hour
    df['Month'] = df['Start Time'].dt.month

    #Drop records with geographically missing features
    df = df.loc[df['Start Station Longitude'] < -65, :]
    df = df.loc[df['Start Station Latitude'] > 35, :]

    #Clean Birth Year, filling NaN's using station means and renaming
    df['Birth Year'] = pd.to_numeric(df['Birth Year'], errors='coerce')
    df['Birth Year'] = df['Birth Year'].apply(lambda x: x if x > 1920 else np.NaN)
    mean_year_by_station = df.groupby('Start Station ID').agg({'Birth Year':np.mean})
    df['Age Missing'] = df['Birth Year'].isnull().astype(int)
    df = pd.merge(df, mean_year_by_station, how='left', left_on='Start Station ID', right_index=True)
    df['Birth Year'] = df['Birth Year_x'].fillna(df['Birth Year_y'])
    df.drop(['Birth Year_x', 'Birth Year_y'], axis=1, inplace=True)
    df['Age'] = df['Start Time'].dt.year - df['Birth Year']
    df.drop('Birth Year',axis=1, inplace=True)

    #Make users dummies
    users = pd.get_dummies(df['User Type'], dummy_na=True)
    users.columns = list(users.columns[0:2]) + ['User Type Missing']
    df = df.join(users)

    df.drop(['Start Station Latitude', 'Start Station Longitude', 'User Type'], axis=1, inplace=True)

    return df

def format_NOAA_data(project_dir):
    """ Formats daily NOAA data by:
            - Filling missing values using the average of forward/back fill
            - Dropping fields with many missing values

        Args:
            -project_dir: Path to project directory (os.path)

        Returns:
            - df: Formatted pandas DataFrame with datetime index

    """
    print "Formatting NOAA data..."
    df = pd.read_csv(os.path.join(project_dir, 'data/external/903571.csv'), na_values=-9999, parse_dates=[2])
    df = df[['DATE','PRCP','SNOW','SNWD','TMAX','TMIN','AWND','WSF2','WSF5', 'WT01', 'WT08']]
    df = df.set_index('DATE')
    df = df.fillna({'WT01':0, 'WT08':0})
    for col in df.columns:
        df[col] = get_forward_back_avg(df[col])
    return df

def get_forward_back_avg(series):
    forward = series.ffill()
    back = series.bfill()
    if np.sum(forward - back) == 0:
        print 'No change for {}'.format(series.name)
    average = (forward + back)/2.0
    return average


