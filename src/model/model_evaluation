import numpy as np
import pandas as pd
import json
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

def load_model(path):

    clf = pickle.load(open(path,'rb'))

    return clf

# import test data
def import_test_data(test_data_path):

    test_data = pd.read_csv(test_data_path)

    return test_data


def process_data(model, test_data):

    X_test = test_data.iloc[:,0:-1].values
    
    y_test = test_data.iloc[:,-1].values

    y_pred = model.predict(X_test)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return y_test, y_pred, y_pred_proba


# Calculate evaluation metrics
def calculate_scores(y_test, y_pred, y_pred_proba):

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_pred_proba)

    metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'auc':auc
    }

    return metrics_dict



def save_metrics(path, metrics_dict):

    with open(path, 'w') as file:
    
        json.dump(metrics_dict, file, indent=4)


def main():

    clf = load_model(path='models/model.pkl')

    test_data = import_test_data(test_data_path='./data/features/test_bow.csv')

    y_test, y_pred, y_pred_proba = process_data(model=clf, test_data=test_data)

    metrics_dict = calculate_scores(y_test, y_pred, y_pred_proba)

    save_metrics(path='reports/metrics.json', metrics_dict=metrics_dict)


if __name__ == '__main__':

    main()