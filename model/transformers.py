# model/transformers.py

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


class DuplicateFeatureFilter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = X.columns
        self.keep_cols_ = X.T.drop_duplicates().T.columns
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return X[self.keep_cols_]


class VarianceFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = X.columns
        self.selector_ = VarianceThreshold(self.threshold)
        self.selector_.fit(X)
        self.keep_cols_ = X.columns[self.selector_.get_support()]
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return X[self.keep_cols_]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.columns_ = X.columns

        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        mi = mutual_info_classif(X, y)
        mi = pd.Series(mi, index=X.columns)

        to_drop = set()

        for c1 in upper.columns:
            for c2 in upper.index:
                if upper.loc[c2, c1] > self.threshold:
                    if mi[c1] < mi[c2]:
                        to_drop.add(c1)
                    else:
                        to_drop.add(c2)

        self.keep_cols_ = [c for c in X.columns if c not in to_drop]
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return X[self.keep_cols_]


class GroupwisePCA(BaseEstimator, TransformerMixin):
    def __init__(self, freq_var=0.9, time_var=0.95, random_state=42):
        self.freq_var = freq_var
        self.time_var = time_var
        self.random_state = random_state

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = X.columns

        group_map = X.columns.to_series().groupby(
            X.columns.map(lambda c: c.split("-")[0].split("(")[0])
        ).apply(list)

        self.group_models_ = []

        for group, cols in group_map.items():
            Xg = X[cols]

            scaler = StandardScaler()
            Xg_scaled = scaler.fit_transform(Xg)

            var = self.freq_var if group.startswith("f") else self.time_var

            pca = PCA(n_components=var, random_state=self.random_state)
            pca.fit(Xg_scaled)

            self.group_models_.append({
                "cols": cols,
                "scaler": scaler,
                "pca": pca
            })

        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)

        out = []
        for g in self.group_models_:
            Xg = g["scaler"].transform(X[g["cols"]])
            out.append(g["pca"].transform(Xg))

        return np.hstack(out)