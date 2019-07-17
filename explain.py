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
    parser.add_argument('--output', metavar='O', nargs='?',
                        default='output/explanations',
                        help='Path to the output directory')
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

y = df[label_column].values
df = df.drop(label_column, axis=1)


path = config['images']
filenames = list(os.walk(path))[0][2]
filenames = np.asarray(sorted(filenames))

keras_path = os.path.join('output', 'explain.h5')

cv = 10
skf = StratifiedKFold(cv, shuffle=True, random_state=42)
for tr_index, ts_index in skf.split(np.arange(df.shape[0]), y):
    # FIXME: these variables have to be adapted to the best performing model
    # for each dataset.
    dw = 5
    du = 3
    dm = 2
    dj = 2

    model = models.joint.JointExplanations(config,
                                        dense_width=dw,
                                        dense_unconstrained_num=du,
                                        dense_monotonic_num=dm,
                                        dense_joint_num=dj,
                                        epochs=10,
                                        path=keras_path,
                                        explanation_completeness=0.75
                                        )

    model.fit(df.iloc[tr_index], y[tr_index])
    preds = model.predict(df.iloc[ts_index])
    explanations = model.explain(df.iloc[ts_index])

    for i, tsi in enumerate(ts_index):
        plt.subplot(131)
        show_img(path, filenames[tsi])
        
        title = 'Image %s (pred %d label %d)' % (filenames[tsi],
                                                 preds[i], y[tsi])
        if not (preds[i] == 1 and y[tsi] == 1):
            continue

        print(i, y[tsi], preds[i])

        for e in explanations[i]:
            if e['type'] == 'neighbor-similar':
                plt.subplot(132)
                show_img(path, filenames[tr_index][e['neighbor']])
                title += '\nSimilar %s: %s' %\
                    (filenames[tr_index][e['neighbor']], e['why'])
            elif e['type'] == 'neighbor-opponent':
                plt.subplot(133)
                show_img(path, filenames[tr_index][e['neighbor']])
                title += '\nOpponent %s: %s' % \
                    (filenames[tr_index][e['neighbor']], e['why'])
            if e['type'] == 'decision-rule-opponent':
                title += '\nRule: %s' % [(x['feature'], x['current-value'],
                                          x['end-point']) for x in e['rule']]

        plt.suptitle(title)
        plt.savefig(os.path.join(args.output, '%s.png' % filenames[tsi]),
                    bbox_inches='tight')
        plt.clf()
