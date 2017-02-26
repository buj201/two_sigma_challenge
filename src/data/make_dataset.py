import itertools
import numpy as np
import pandas as pd
import os
from ..features import build_features

def main(project_dir):

    """ Runs all data processing and cleaning, turning raw and external data
        (from data/raw and data/external) into final tidy data frame for
        modeling (saved in data/processed)/.

        Args:
            -project_dir: Path to project directory (os.path)

        Returns:
            - None (data saved in data/processed)

    """
    #Read in and clean trips data and weather data
    NOAA = build_features.format_NOAA_data(project_dir)
    all_trips = join_monthly_data(project_dir)

    #Merge datasets on date
    all_trips['DATE'] = all_trips['Start Time'].dt.floor('d')
    merged = pd.merge(df, weather, how='left', on='DATE')
    for feature, count in merged.count().iteritems():
        assert count == merged.shape[0], '{} is missing {} values.'.format(feature, merged.shape[0] - count)

    final_path = os.path.join(project_dir, 'data/processed/final_dataset.csv')
    merged.to_csv(final_path, index = False, compression='gzip')


def join_monthly_data(project_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned trip data, which is returned (as Pandas DataFrame) for
        merging with NOAA data.

        Args:
            -project_dir: Path to project directory (os.path)

        Return:
            - raw_data: Pandas DataFrame
    """

    #Note the columns numbers are consistant, but names are not
    columns = [0,1,3,5,6,12,13,14]

    column_names = ['Trip Duration',
           'Start Time',
           'Start Station ID',
           'Start Station Latitude',
           'Start Station Longitude',
           'User Type',
           'Birth Year',
           'Gender']

    first = True

    years = [2016]
    months = [str(x).zfill(2) for x in range(1,13)]

    for year, month in itertools.product(years, months):
        print "Reading data for {}/{}...".format(month, year)
        filename = os.path.join(project_dir, 'data/raw/{}{}-citibike-tripdata.zip'.format(year, month))
        df = pd.read_csv(filename, parse_dates=[1], usecols=columns)

        df.columns = column_names

        df = build_features.format_monthly_data(df)
        if first:
            all_data = df
            first = False
        else:
            all_data = all_data.append(df, ignore_index=True)

    return all_data


if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    main(project_dir)
