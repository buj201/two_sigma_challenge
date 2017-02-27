import numpy as np
import pandas as pd
import pickle
import os

#Models
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

#Preprocessing
from sklearn.preprocessing import OneHotEncoder

#Model Selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV


def main(project_dir):
    """ Train ElasticNet linear model and GBR model using
        grid search over a fixed validation set. Note the
        models are pickled and saved in the models/
        subdirectory.

        Args:
            -project_dir: top level directory for project

        Returns:
            -None- dumps pickled models to models/ for
            later evaluation, visualization, and prediction.

    """

    categorical_features = ['Start Station ID',
                            'Gender',
                            'Day of Week',
                            'Hour',
                            'Month',
                            'Age Missing',
                            'Customer',
                            'Subscriber',
                            'User Type Missing',
                            'WT01',
                            'WT08']

    print 'Reading data...'
    data_path = os.path.join(project_dir,'data/processed/train.csv.gz')
    df = pd.read_csv(data_path)

    X = df.drop(['Trip Duration'], axis=1)
    y = df['Trip Duration']

    mask = np.zeros(X.shape[1])
    for i in range(len(mask)):
        if X.columns[i] in categorical_features:
            mask[i] = 1
        else:
            mask[i] = 0
    mask = mask.astype(bool)

    print 'Sparsifying categorical features...'
    enc = OneHotEncoder(categorical_features=mask)
    X = enc.fit_transform(X)

    np.random.seed(1354)
    validation = PredefinedSplit(np.random.choice([0,-1],X.shape[0],p=[0.8,0.2]))

    print 'Running Elastic Net Grid Search...'

    EN_params = [{
        'alpha':[10,1,0.1, 0.01,0.001],
        'l1_ratio':[0.0, 0.5, 1.0]
    }]

    EN_grid_search = GridSearchCV(ElasticNet(fit_intercept=True, normalize=True, selection='random', random_state=24351), EN_params, cv=validation, n_jobs=-1, verbose=2)
    EN_grid_search.fit(X, y)

    print 'Pickling and saving Elastic Net models...'

    with open(os.path.join(project_dir,'models/elastic_net.pkl'), 'wb') as outfile:
        pickle.dump(EN_grid_search, outfile)

    print 'Running GBR Grid Search...'

    GBR_params = [{
        'learning_rate':[0.01,0.1],
        'max_depth':[2,4,6]
    }]

    GBR_grid_search = GridSearchCV(GradientBoostingRegressor(subsample=0.8,verbose=2, random_state=24351, n_estimators=400), GBR_params, cv=validation, n_jobs=-1, sverbose=2)
    GBR_grid_search.fit(X, y)

    print 'Pickling and saving GBR models...'

    with open(os.path.join(project_dir,'models/GBR.pkl'), 'wb') as outfile:
        pickle.dump(EN_grid_search, outfile)

if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    main(project_dir)
