import numpy as np
import pandas as pd
import os

#Preprocessing
from sklearn.preprocessing import StandardScaler

#Model Selection
from sklearn.model_selection import train_test_split


def main(project_dir):
    """ Makes training/test splits and saves to data/processed.

    """

    categorical_features = ['Start Station ID',
                            'Gender',
                            'Day of Week',
                            'Hour',
                            'Month',
                            'Age Missing',
                            'User Type'
                            'WT01',
                            'WT08']

    continuous_features = ['Age',
                           'PRCP',
                           'SNOW',
                           'SNWD',
                           'TMAX',
                           'TMIN',
                           'AWND',
                           'WSF2',
                           'WSF5']

    print 'Reading data...'
    data_path = os.path.join(project_dir,'data/processed/final_dataset.csv.gz')
    df = pd.read_csv(data_path)

    df = df.drop(['Start Time','DATE'], axis=1)

    print 'Formatting categorical features...'
    for feature in categorical_features:
        df[feature] = map_cat_feature_to_target_range(df[feature])

    print 'Splitting into training/validation and test data...'
    df_train, df_test = train_test_split(df, test_size = 0.2, random_state=85645)

    print 'Scaling continuous features...'
    sc = StandardScaler()
    df_train.loc[:,continuous_features] = sc.fit_transform(df_train[continuous_features])
    df_test.loc[:,continuous_features] = sc.transform(df_test[continuous_features])

    print 'Saving train/test data...'
    train_path = os.path.join(project_dir,'data/processed/train.csv.gz')
    df_train.to_csv(train_path, index=False, compression='gzip')
    test_path = os.path.join(project_dir,'data/processed/test.csv.gz')
    df_test.to_csv(test_path, index=False, compression='gzip')

def map_cat_feature_to_target_range(series):
    vals = series.unique()
    mapping = dict(zip(vals,range(len(vals))))
    return series.apply(lambda x: mapping[x])

if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    main(project_dir)
