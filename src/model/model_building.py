import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml

def load_params(params_path):

    params = yaml.safe_load(open(params_path, 'r'))['model_training']

    return params

# import data
def import_data(train_data_path):

    train_df = pd.read_csv(train_data_path)
    
    return train_df

def split_data(train_df):

    X_train = train_df.iloc[:, 0:-1].values

    y_train = train_df.iloc[:, -1].values

    return X_train, y_train


# define and train model
def train_model(params, X_train, y_train):

    clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])

    clf.fit(X_train, y_train)

    return clf


# save model
def save_model(model, path):

    pickle.dump(model, open(path,'wb'))

def main():
    params = load_params('params.yaml')

    train_df = import_data('data/features/train_bow.csv')

    X_train, y_train = split_data(train_df)

    clf = train_model(params, X_train, y_train)

    path = 'models/model.pkl'

    save_model(clf, path)


if __name__ == '__main__':

    main()  
