import numpy as np
import pandas as pd
import pickle

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

    data_path = os.path.join(project_dir,'data/processed/final_dataset.csv'))
    df = pd.read_csv(data_path)

    X = df.drop(['Trip Time','Start Time','TIME'], axis=1)
    y = df['Trip Time']

    enc = OneHotEncoder(categorical_features=categorical_features)
    X = enc.fit_transform(X)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = 0.2)

    np.random.seed(1354)
    validation = PredefinedSplit(np.random.choice([0,-1],X_train_val.shape[0],p=[0.8,0.2]))

    EN_params = [{
        'alpha':np.logspace(-6,2,5),
        'l1_ratio':np.linspace(0,1,4)
    }]

    EN_grid_search = GridSearchCV(ElasticNet(fit_intercept=True, normalize=True, selection='random', random_state=24351), EN_params, cv=validation)
    EN_grid_search.fit(X_train_val, y_train_val)

    with open(os.path.join(project_dir,'models/elastic_net.pkl'), 'wb') as outfile:
        pickle.dump(EN_grid_search, outfile)

    GBR_params = [{
        'learning_rate':[0.01,0.1],
        'max_depth':[2,4,6]
    }]

    GBR_grid_search = GridSearchCV(GradientBoostingRegressor(subsample=0.8,verbose=2, random_state=24351, n_estimators=400), GBR_params, cv=validation)
    GBR_grid_search.fit(X_train_val, y_train_val)

    with open(os.path.join(project_dir,'models/GBR.pkl'), 'wb') as outfile:
        pickle.dump(EN_grid_search, outfile)

def map_cat_feature_to_target_range(series):
    vals = series.unique()
    mapping = dict(zip(vals,range(len(vals))))
    return series.apply(lambda x: mapping[x])

if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    main(project_dir)
