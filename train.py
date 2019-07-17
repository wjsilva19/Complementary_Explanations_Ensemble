import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import argparse
import json
import cv2
import os

import models.joint


def get_args():
    parser = argparse.ArgumentParser(description='Train the models.')
    parser.add_argument('--data', metavar='D', nargs='?',
                        help='Path to the dataset')
    parser.add_argument('--config', metavar='CF', nargs='?',
                        help='Path to the configuration file')
    parser.add_argument('--constraints', metavar='C', type=int, default=1,
                        help='Use constraints (0-1)')
    parser.add_argument('--train', metavar='T', type=int, default=1,
                        help='Train (0-1)')
    args = parser.parse_args()
    return args


def show_img(path, filename):
    img = cv2.imread(os.path.join(path, filename))
    plt.imshow(img[:, :, ::-1])


args = get_args()

df = pd.read_csv(args.data)

f = open(args.config, 'r')
config = json.load(f)
f.close()

label_column = config["target"]
monotonic_fts = config["monotonic"]
unconstrained_fts = config["unconstrained"]

y = df[label_column].values
df = df.drop(label_column, axis=1)

if args.constraints == 0:
    unconstrained_fts = unconstrained_fts + monotonic_fts
    monotonic_fts = []

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

# Correctness: 100%
# Completeness: 100%
# Length: 2

#preds = cross_val_predict(DecisionTreeClassifier(max_depth=None), df.as_matrix(), y, cv=10)
#print(accuracy_score(y, preds))

#model = models.joint.JointExplanations(config)

#model = models.NeighborTripletLossExplanation(unconstrained_features=unconstrained_fts,
                                              #monotonic_features=monotonic_fts)

if args.train != 0:
    model.fit(df, y)
model.load_model()

preds = model.predict(df)
explanations = model.explain(df)

path = config['images']
filenames = list(os.walk(path))[0][2]
filenames = sorted(filenames)

e_types = ['probability', 'neighbor-similar', 'neighbor-opponent',
           'decision-rule']

res_type = {t: {'accuracy': [], 'completeness': []}
                    for t in e_types}

for i in range(len(df)):
    print(i, y[i], preds[i])
    plt.subplot(131)
    show_img(path, filenames[i])

    for e in explanations[i]:
        if e['type'] == 'probability':
            print('Probability', e['value'])
        elif e['type'] == 'neighbor-similar':
            #plt.subplot(132)
            #show_img(path, filenames[e['neighbor']])
            print('Similar', e['why'])
        elif e['type'] == 'neighbor-opponent':
            #plt.subplot(133)
            #show_img(path, filenames[e['neighbor']])
            print('Opponent', e['why'])
        elif e['type'] == 'decision-rule':
            print(e['rule'])
        else:
            print(e)

        print('Coverage:')
        e_type = e['type']
        cov = e['coverage']
        if len(cov) > 0:
            acc = metrics.accuracy_score(y[cov], preds[cov])
        else:
            acc = 1.

        res_type[e_type]['accuracy'].append(acc)
        res_type[e_type]['completeness'].append(len(cov) / len(df))
        
        print(' Accuracy: %.4f' % np.mean(res_type[e_type]['accuracy']))
        print(' Completeness: %.4f' %
              np.mean(res_type[e_type]['completeness']))

    #plt.show()
    print()

print('Random: %.4f' % max(np.mean(y), 1 - np.mean(y)))
print('Accuracy: %.4f' % metrics.accuracy_score(y, model.predict(df)))
print('ROC AUC: %.4f' % metrics.roc_auc_score(y, model.predict_proba(df)[:,
                                                                         1]))
print('F1: %.4f' % metrics.f1_score(y, model.predict(df)))
print()

for t in e_types:
    print(t)
    print('Accuracy: %.4f' % np.mean(res_type[t]['accuracy']))
    print('Completeness: %.4f' % np.mean(res_type[t]['completeness']))
    print()

"""
TO DO:
- Compression metric
- Decision tree explanations
- Scorecard explanations
- cross-validation: leave-one-out
- model selection: K-fold
"""
