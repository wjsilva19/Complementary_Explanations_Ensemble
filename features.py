from sklearn.base import BaseEstimator, TransformerMixin


class OrdinalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features = []

    def fit(self, df, y):
        return self

    def transform(self, df):
        return df.as_matrix()


class MissingValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features = []

    def fit(self, df, y):
        return self

    def transform(self, df):
        return df.as_matrix()
