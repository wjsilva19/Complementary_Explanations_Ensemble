from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np


class Tree:
    def __init__(self, df, y, config, lower, upper, max_depth):
        self.config = config
        self.lower = lower
        self.upper = upper
        self.EPS = 1e-25

        self.pred = y.mean()

        proceed = True

        if self.pred < lower or self.pred > upper:
            self.pred = (lower + upper) / 2.
        else:
            self.pred = y.mean()

        if lower > 0.5:
            proceed = False
        elif upper < 0.5:
            proceed = False

        self.is_leaf = True

        if proceed and max_depth > 0 and df.shape[0] > 0 and len(set(y)) > 1:
            splits = self.get_splits(df, y)
            if len(splits) == 0:
                return

            self.is_leaf = False

            split = sorted(splits, key=lambda x: self.entropy(df, y, x))[0]
            self.feature = split[0]
            self.threshold = split[1]

            values = df[self.feature].values
            mask = values <= self.threshold

            if self.feature in self.config['monotonic']:
                mid_point = (split[2] + split[3]) / 2.
                mid_point = max(lower, mid_point)
                mid_point = min(mid_point, upper)

                self.left = Tree(df.loc[mask], y[mask], config,
                                lower, mid_point,
                                max_depth - 1)
                self.right = Tree(df.loc[~mask], y[~mask], config,
                                mid_point, upper,
                                max_depth - 1)
            else:
                self.left = Tree(df.loc[mask], y[mask], config,
                                 lower, upper, max_depth - 1)
                self.right = Tree(df.loc[~mask], y[~mask], config,
                                  lower, upper, max_depth - 1)


    def entropy(self, df, y, split):
        values = df[split[0]].values

        left = y[values <= split[1]].mean()
        right = y[values > split[1]].mean()

        eleft = left * np.log2(left + self.EPS) + \
            (1 - left) * np.log2(1 - left+ self.EPS)
        eright = right * np.log2(right + self.EPS) + \
            (1 - right) * np.log2(1 - right + self.EPS)

        return -eleft - eright

    def get_splits(self, df, y):
        ret = []

        for f in self.config['unconstrained']:
            all_ = df[f].values

            values = list(sorted(zip(df[f].values, y)))
            values = [(v0 + v1) / 2.
                      for (v0, y0), (v1, y1) in zip(values, values[1:])
                      if y0 != y1 and v0 != v1]
            values = [(f, v, None, None) for v in values]

            ret += values

        for f in self.config['monotonic']:
            all_ = df[f].values

            values = list(sorted(zip(df[f].values, y)))
            values = [(v0 + v1) / 2.
                      for (v0, y0), (v1, y1) in zip(values, values[1:])
                      if y0 != y1 and v0 != v1]

            values = [(f, v, y[all_ <= v].mean(), y[all_ > v].mean())
                      for v in values
                      if y[all_ <= v].mean() < y[all_ > v].mean()]
            ret += values

        return ret

    def predict(self, df):
        if self.is_leaf:
            return np.ones(df.shape[0]) * self.pred
        else:
            values = df[self.feature].values
            mask = values <= self.threshold

            ret = np.zeros(df.shape[0])
            ret[mask] = self.left.predict(df.loc[mask])
            ret[~mask] = self.right.predict(df.loc[~mask])

            return ret


class MonotonicDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, config, max_depth=None):
        self.config = config

        # Features
        self.all_features = config["unconstrained"] + config["monotonic"]

        self.max_depth = max_depth

    def fit(self, df, y):
        max_depth = self.max_depth if self.max_depth is not None else np.inf
        self.tree = Tree(df, y, self.config, 0., 1., max_depth)
        return self

    def predict(self, df):
        return self.predict_proba(df)[:, 1],round()

    def predict_proba(self, df):
        probs = self.tree.predict(df)[:, np.newaxis]
        return np.hstack((1 - probs, probs))


import matplotlib.pyplot as plt
import cv2

img = cv2.imread('tree.bmp', 0)
img = cv2.resize(img, None, fx=0.25, fy=0.25) > 128
img = img[::-1, :]

N = np.prod(img.shape)

X = np.indices(img.shape).astype(np.float)
X[0] = X[0] / img.shape[0]
X[1] = X[1] / img.shape[1]

X = np.asarray(list(zip(X[0].ravel(), X[1].ravel())))
X = pd.DataFrame(X, columns=['f0', 'f1'])

y = img.reshape(N)

config = {'unconstrained': [], 'monotonic': ['f0', 'f1']}
tree = MonotonicDecisionTreeClassifier(config, max_depth=None)
tree.fit(X, y)

probs = tree.predict_proba(X)[:, 1].reshape(img.shape)

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(probs, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(probs.round(), cmap='gray')
plt.savefig('monotonic-tree.png', bbox_inches='tight')
plt.show()
