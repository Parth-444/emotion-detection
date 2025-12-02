import numpy as np
import pandas as pd
import yaml
import os
from sklearn.feature_extraction.text import CountVectorizer

def load_params(params_path):

    max_features = yaml.safe_load(open(params_path, 'r'))['feature_engineering']['max_features']
    
    return max_features


# fetching data from processed
def load_data(train_path, test_path):

    train_data = pd.read_csv(train_path)

    test_data = pd.read_csv(test_path)

    return train_data, test_data

# apply bow
def process_data(train_data, test_data):

    train_data.fillna('',inplace=True)

    test_data.fillna('',inplace=True)

    X_train = train_data['content'].values

    y_train = train_data['sentiment'].values

    X_test = test_data['content'].values
    
    y_test = test_data['sentiment'].values

    return X_train, y_train, X_test, y_test



def load_model(max_features):
    vectorizer = CountVectorizer(max_features=max_features)
    return vectorizer    


def apply_bow(vectorizer, X_train, y_train, X_test, y_test):

    X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
    X_test_bow = vectorizer.transform(X_test)

    train_df = pd.DataFrame(X_train_bow.toarray())

    train_df['label'] = y_train

    test_df = pd.DataFrame(X_test_bow.toarray())

    test_df['label'] = y_test

    return train_df, test_df


# store the data

def store_data(data_path, train_df, test_df):

    os.makedirs(data_path)   

    train_df.to_csv(os.path.join(data_path, "train_bow.csv"))

    test_df.to_csv(os.path.join(data_path, "test_bow.csv"))


def main():

    max_features = load_params('./params.yaml')

    train_data, test_data = load_data(train_path='./data/processed/train_processed_data.csv', test_path='./data/processed/test_processed_data.csv')

    X_train, y_train, X_test, y_test = process_data(train_data, test_data)

    vectorizer = load_model(max_features=max_features)

    train_df, test_df = apply_bow(vectorizer=vectorizer, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    data_path = os.path.join('data', 'features')
    
    store_data(data_path=data_path, train_df=train_df, test_df=test_df)

if __name__ == '__main__':

    main()