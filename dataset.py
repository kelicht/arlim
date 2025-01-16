import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



class Dataset():

    def __init__(
        self, 
        X, 
        y, 
        feature_names=[], 
        is_binary=[],
        is_immutable=[], 
        cf_size=0.25, 
        round_size=0.25,
        is_scaling=True,
        age_index=-1,
    ):
        self.X = X
        self.y = y
        self.feature_names = feature_names if len(feature_names) == X.shape[1] else ['x_{}'.format(d) for d in range(X.shape[1])]
        self.is_binary = is_binary if len(is_binary) == X.shape[1] else [False] * X.shape[1]
        self.is_immutable = is_immutable if len(is_immutable) == X.shape[1] else [False] * X.shape[1]
        self.cf_size = cf_size
        self.round_size = round_size
        self.is_scaling = is_scaling
        self.age_index = age_index


    def initialize(self):
        X, y = self.X.copy(), self.y.copy()
        if self.is_scaling: X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        test_size = self.cf_size + self.round_size
        self.X_tr, X_remain, self.y_tr, y_remain = train_test_split(X, y, test_size=test_size, stratify=y)
        test_size = self.round_size / test_size
        self.X_cf, self.X_ts, self.y_cf, self.y_ts = train_test_split(X_remain, y_remain, test_size=test_size, stratify=y_remain)     
        return self   

    
    def getTrainingSamples(self):
        return self.X_tr, self.y_tr


    def getCounterfactuals(self, estimator):
        is_valid = (estimator.predict(self.X_cf) == 1)
        return self.X_cf[is_valid], self.y_cf[is_valid]

    
    def getInstances(self, estimator):
        is_target = (estimator.predict(self.X_ts) == 0) * (self.y_ts == 0)
        return self.X_ts[is_target]

    
    def getIsYounger(self, X):
        if X.ndim == 1:
            return (X[self.age_index] == 1)
        else:
            return (X[:, self.age_index] == 1)

    

class CreditDataset(Dataset):

    def __init__(
        self,
        cf_size=0.25, 
        round_size=0.25,
        is_scaling=True,
        filepath='./datasets/credit.csv',
    ):
        df = pd.read_csv(filepath)
        y = df.pop('DefaultNextMonth').values
        feature_names = list(df.columns)
        X = df.values
        is_binary = [True]*3 + [False]*10
        is_immutable = [True]*3 + [False]*10
        age_index = 2
        super().__init__(X, y, feature_names, is_binary, is_immutable, cf_size, round_size, is_scaling, age_index)
        self.name = 'Credit'



class CompasDataset(Dataset):

    def __init__(
        self,
        cf_size=0.25, 
        round_size=0.25,
        is_scaling=True,
        filepath='./datasets/compas.csv',
    ):
        df = pd.read_csv(filepath)
        y = df.pop('RecidivateWithinTwoYears').values
        feature_names = list(df.columns)
        X = df.values
        is_binary = [False]*4 + [True]*5
        is_immutable = [False]*6 + [True]*3
        age_index = -3
        super().__init__(X, y, feature_names, is_binary, is_immutable, cf_size, round_size, is_scaling, age_index)
        self.name = 'COMPAS'
        


class DiabetesDataset(Dataset):

    def __init__(
        self,
        cf_size=0.25, 
        round_size=0.25,
        is_scaling=True,
        filepath='./datasets/diabetes.csv',
    ):
        df = pd.read_csv(filepath)
        y = df.pop('Outcome').values
        feature_names = list(df.columns)
        X = df.values
        is_binary = [True] + [False]*6 + [True]
        is_immutable = [True] + [False]*6 + [True]
        age_index = -1
        super().__init__(X, y, feature_names, is_binary, is_immutable, cf_size, round_size, is_scaling, age_index)
        self.name = 'Diabetes'
        
        
        
class GermanDataset(Dataset):

    def __init__(
        self,
        cf_size=0.25, 
        round_size=0.25,
        is_scaling=True,
        filepath='./datasets/german.csv',
    ):
        df = pd.read_csv(filepath)
        y = df.pop('GoodCustomer').values
        feature_names = list(df.columns)
        X = df.values
        is_binary = [True]*2 + [False]*6 + [True]*3
        is_immutable = [True]*2 + [False]*9
        age_index = 0
        super().__init__(X, y, feature_names, is_binary, is_immutable, cf_size, round_size, is_scaling, age_index)
        self.name = 'German'



class SyntheticDataset(Dataset):

    def __init__(
        self,
        cf_size=0.25, 
        round_size=0.25,
        is_scaling=True,
        filepath='./datasets/synthetic.csv',
    ):
        df = pd.read_csv(filepath)
        y = df.pop('Accept').values
        feature_names = list(df.columns)
        X = df.values
        is_binary = [False]*3 + [True]
        is_immutable = [False]*3 + [True]
        age_index = -1
        super().__init__(X, y, feature_names, is_binary, is_immutable, cf_size, round_size, is_scaling, age_index)
        self.name = 'Synthetic'
        