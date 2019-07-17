from sklearn.model_selection import cross_val_predict, StratifiedKFold, \
    train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import json
import cv2
import os

from models import DecisionTreeExplanation, NeighborExplanation
import models.joint

np.random.seed(42)
tf.set_random_seed(42)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


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


def pandas_cross_val_predict(clf, df, y, cv=10, method='predict'):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    index = np.arange(df.shape[0])

    preds = np.zeros(df.shape[0], dtype=np.float
                     if method != 'predict' else np.int)

    for tr_index, ts_index in skf.split(index, y):
        tr_df = df.iloc[tr_index]
        tr_y = y[tr_index]

        ts_df = df.iloc[ts_index]
        
        clf.fit(tr_df, tr_y)
        if method == 'predict':
            next_preds = clf.predict(ts_df)
        else:
            next_preds = clf.predict_proba(ts_df)[:, 1]
        preds[ts_index] = next_preds
    return preds


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
all_features = unconstrained_fts, monotonic_fts

keras_path = os.path.join('output',
                          args.data.split('/')[-1].replace('.csv',
                                                           '.h5'))

models = [('Tree', [DecisionTreeExplanation(config, criterion=criterion,
                                            max_depth=depth)
                    for depth in [1, 2, 3, 5, 7, 10, None]
                    for criterion in ['gini', 'entropy']
                    ]),
          ('KNN', [NeighborExplanation(config, neighbors=n)
                   for n in [1,]
                   for w in ['uniform',]
                   ]),
          ('DNN', [models.joint.JointExplanations(config,
                                                  dense_width=dw,
                                                  dense_unconstrained_num=du,
                                                  dense_monotonic_num=dm,
                                                  dense_joint_num=dj,
                                                  epochs=1000,
                                                  path=keras_path,
                                                  )
                   for dw in [5, 10]
                   for du in [1, 2, 3]
                   for dm in [1, 2,]
                   for dj in [1, 2,]
                   if du >= dm])
          ]

best_models = []

cv = 10

for group_id, (group_name, group) in enumerate(models):
    best_model = None
    best_score = -np.inf

    print(group_name, len(group))

    for model_id, model in enumerate(group):
        print(model_id, model)
        preds = pandas_cross_val_predict(model, df, y, cv, 'predict_proba')

        score = metrics.roc_auc_score(y, preds)

        if score > best_score:
            best_model = model_id
            best_score = score
            print(group_name, group[best_model], score)
        print()

    best_models.append((group_name, group[best_model]))

clfs = best_models

skf = StratifiedKFold(cv, shuffle=True, random_state=42)

path = config['images']
filenames = list(os.walk(path))[0][2]
filenames = np.asarray(sorted(filenames))

e_types = ['probability', 'neighbor-similar', 'neighbor-opponent',
           'decision-rule-boundary', 'decision-rule-opponent',
           'tree-rule',
           'euclidean-neighbor-similar',
           'euclidean-neighbor-opponent']

"""
dnn = models.joint.JointExplanations(config, epochs=100)
dec_tree = DecisionTreeExplanation(config, criterion='entropy',
                                    max_depth=3)

knn = NeighborExplanation(config)
clfs = [('DecTree', dec_tree), ('DNN', dnn)]
clfs = [('DNN', dnn),
        ('DecTree', dec_tree),
        ('KNN', knn)
        ]
"""

clf_results = {name: {'accuracy': [], 'auc': [], 'pr': [], 'f1': []}
               for name, _ in clfs}

res_type = {t: {'accuracy': [], 'completeness': [], 'compactness': []}
            for t in e_types}

print('Random: %.4f' % max(np.mean(y), 1 - np.mean(y)))

for foldid, (tr_index, ts_index) in enumerate(skf.split(np.arange(df.shape[0]),
                                                        y)):

    print('Fold %d' % foldid)

    for name, model in clfs:
        model.fit(df.iloc[tr_index], y[tr_index])

        tr_preds = model.predict(df.iloc[tr_index])

        preds = model.predict(df.iloc[ts_index])
        probs = model.predict_proba(df.iloc[ts_index])[:, 1]

        #print(y[ts_index])
        #print(preds)

        explanations = model.explain(df.iloc[ts_index])
        acc = metrics.accuracy_score(y[ts_index], preds)
        auc = metrics.roc_auc_score(y[ts_index], probs)
        pr = metrics.average_precision_score(y[ts_index], probs)
        f1 = metrics.f1_score(y[ts_index], preds)

        clf_results[name]['accuracy'].append(acc)
        clf_results[name]['auc'].append(auc)
        clf_results[name]['pr'].append(pr)
        clf_results[name]['f1'].append(f1)

        print(name)
        print('Accuracy: %.4f' % np.mean(clf_results[name]['accuracy']))
        print('ROC AUC: %.4f' % np.mean(clf_results[name]['auc']))
        print('PR AUC: %.4f' % np.mean(clf_results[name]['pr']))
        print('F1: %.4f' % np.mean(clf_results[name]['f1']))
        print()

        for i in range(len(df.iloc[ts_index])):
            #print(i, y[ts_index][i], preds[i])
            #plt.subplot(131)
            #show_img(path, filenames[ts_index][i])

            for e in explanations[i]:
                """
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
                elif e['type'] == 'tree-rule':
                    print(e['rule'])
                """

                #print('Coverage:')
                e_type = e['type']
                cov = e['coverage']

                if len(cov) > 0:
                    acc = metrics.accuracy_score(y[tr_index][cov],
                                                 [preds[i]] *
                                                 len(y[tr_index][cov]))
                else:
                    acc = 1.

                res_type[e_type]['accuracy'].append(acc)
                res_type[e_type]['completeness'].append(len(cov) / len(tr_index))
                res_type[e_type]['compactness'].append(e.get('compactness',
                                                             np.inf))

                #print(' Accuracy: %.4f' % np.mean(res_type[e_type]['accuracy']))
                #print(' Completeness: %.4f' %
                    #np.mean(res_type[e_type]['completeness']))

            #plt.show()
            #print()

    for t in e_types:
        print(t)
        print('Accuracy: %.4f' % np.mean(res_type[t]['accuracy']))
        print('Completeness: %.4f' % np.mean(res_type[t]['completeness']))
        print('Compactness: %.4f' % np.mean(res_type[t]['compactness']))
        print()

"""
TO DO:
- Compression metric
- Decision tree explanations
- Scorecard explanations
- cross-validation: leave-one-out
- model selection: K-fold
"""
