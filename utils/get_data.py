import numpy as np
from sklearn.datasets import load_breast_cancer

def get_breast_cancer_data():
    cancer = load_breast_cancer()
    x = cancer['data']
    y = cancer['target']
    feature_names = cancer['feature_names']
    data = np.concatenate((x, y.reshape(-1, 1)), axis=1)
    return {
        'data' : data,
        'feature_names' : feature_names
    }

def get_data(name):
    if name == 'cancer':
        data = get_breast_cancer_data()

    return data
