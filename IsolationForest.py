import numpy as np
import pandas as pd


class IsolationTree:

    def __init__(self, height_limit, current_height=0):
        self.height_limit = height_limit
        self.current_height = current_height
        self.split_by = None
        self.split_value = None
        self.left = None
        self.right = None
    
    def fit(self, X):
        if self.current_height >= self.height_limit or X.shape[0] <= 1:
            return self
        n_samples, n_features = X.shape
        self.split_by = np.random.choice(n_features)
        self.split_value = np.random.uniform(X[:, self.split_by].min(), X[:, self.split_by].max())
        left = X[X[:, self.split_by] < self.split_value]
        right = X[X[:, self.split_by] >= self.split_value]
        # print(left.shape, right.shape, X.shape, self.current_height, self.height_limit)
        self.left = IsolationTree(self.height_limit, self.current_height + 1).fit(left)
        self.right = IsolationTree(self.height_limit, self.current_height + 1).fit(right)
        return self

    def predict(self, X):
        if self.current_height >= self.height_limit:
            return np.ones(X.shape[0])
        return (X[:, self.split_by] < self.split_value) * self.left.predict(X) + (X[:, self.split_by] >= self.split_value) * self.right.predict(X)
    
    def path_length(self, X, current_height=0):
        # print(X.shape, self.current_height, self.height_limit)
        if self.current_height >= self.height_limit or X.shape[0] <= 1 or self.split_value is None:
            return self.current_height
        print(self.split_value, self.current_height, X.shape)
        return current_height + (X[:, self.split_by] < self.split_value) * self.left.path_length(X, current_height + 1) + (X[:, self.split_by] >= self.split_value) * self.right.path_length(X, current_height + 1)


class IsolationForest:

    def __init__(self, n_trees=100, height_limit=10):
        self.n_trees = n_trees
        self.height_limit = height_limit
        self.trees = []

    def fit(self, X):
        self.trees = [IsolationTree(self.height_limit).fit(X) for _ in range(self.n_trees)]
        return self
    
    def predict(self, X):
        return np.mean([tree.predict(X) for tree in self.trees], axis=0)
    
    def path_length(self, X):
        return np.mean([tree.path_length(X) for tree in self.trees], axis=0)
    

if __name__ == '__main__':
    df = pd.read_csv('cancer.csv')
    X = df.drop(['diagnosis'], axis=1).values
    y = df['diagnosis'].values
    iforest = IsolationForest(n_trees=100, height_limit=10).fit(X)
    print(iforest.path_length(X))