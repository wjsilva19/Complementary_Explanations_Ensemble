from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import RobustScaler, Imputer
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import mean_squared_error, mean_absolute_error
from keras.constraints import non_neg
from keras.regularizers import l1, l2
from keras import layers as KL
from keras import backend as K
from keras.models import Model

import tensorflow as tf

from scipy.stats import rankdata, norm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import zlib
import cv2
import os


np.random.seed(42)
tf.set_random_seed(42)


def float_to_str(i):
    ret = []
    value = np.asarray(i, dtype=np.float32).view(np.int32).item()
    while value != 0:
        res = value % 256
        ret.append(chr(res))
        value = value // 256

    return ''.join(ret)


def triplet_loss(_, y_pred):
    alpha = 0  # 0.2
    return K.mean(K.maximum(0., y_pred + alpha ))


def triplet_correctness(_, y_pred):
    return K.mean(y_pred <= 0)


class JointExplanations(BaseEstimator, ClassifierMixin):
    def __init__(self, config,
                 dense_width=10,
                 dense_unconstrained_num=3,
                 dense_monotonic_num=2,
                 dense_joint_num=2,
                 dense_activation='tanh',
                 dropout=0.5, l2=0, l1=1e-5,
                 explanation_completeness=0.95,
                 include_coverage=True,
                 path='model.h5', epochs=1000, batch_size=64):

        self.config = config

        # Features
        self.unconstrained_features = config["unconstrained"]
        self.monotonic_features = config["monotonic"]

        self.all_features = self.unconstrained_features + \
            self.monotonic_features
        self.num_features = len(self.all_features)

        self.num_monotonic = len(self.monotonic_features)
        self.num_unconstrained = len(self.unconstrained_features)

        # Architecture
        self.dense_width = dense_width
        self.dense_unconstrained_num = dense_unconstrained_num
        self.dense_monotonic_num = dense_monotonic_num
        self.dense_joint_num = dense_joint_num
        self.dense_activation = dense_activation

        # Regularization
        self.l2 = l2
        self.l1 = l1
        self.dropout = dropout

        # Utils
        self.path = path
        self.batch_size = batch_size
        self.epochs = epochs

        # Explanation
        self.explanation_completeness = explanation_completeness
        self.include_coverage = include_coverage

    def fit(self, df, y):
        self.knowledge_df = df.copy()
        self.knowledge_y = y.copy()

        #self.load_model()
        self.model = self.build_model()

        # Create tuples
        df_ = df[self.all_features].as_matrix().astype(np.float)
        
        #df_ = (df_ - df_.min(axis=0, keepdims=True)) / \
            #(df_.max(axis=0, keepdims=True) - df_.min(axis=0, keepdims=True) + 1e-15)
        #print(df_.max(axis=0, keepdims=True))

        index = np.arange(df.shape[0])
        train_index, val_index, _, _ = train_test_split(index, y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=42)

        #tr_anchor, tr_same, tr_diff, tr_labels = \
        #    self.build_triplets(df_[train_index], y[train_index])
        #val_anchor, val_same, val_diff, val_labels = \
        #    self.build_triplets(df_[val_index], y[val_index])

        self.classification_model.fit(df_[train_index], y[train_index],
                                      validation_data=(df_[val_index],
                                                       y[val_index]),
                                      callbacks=[ModelCheckpoint(self.path,
                                                                 save_best_only=True,
                                                                 verbose=1),
                                                 EarlyStopping(patience=10), #FIXME
                                                 ],
                                      epochs=self.epochs,
                                      verbose=2,
                                      batch_size=256,
                                      shuffle=True)
        self.classification_model.load_weights(self.path)

        #print(tr_labels.mean(), val_labels.mean())

        """
        self.model.fit({'anchor': tr_anchor,
                        'positive': tr_same,
                        'negative': tr_diff},
                       {'output-diff': tr_labels,
                        'output-prob': tr_labels},
                       validation_data = ({'anchor': val_anchor,
                                           'positive': val_same,
                                           'negative': val_diff},
                                          {'output-diff': val_labels,
                                           'output-prob': val_labels}
                                          ),
                       callbacks=[ModelCheckpoint(self.path,
                                                  save_best_only=True,
                                                  verbose=1),
                                  EarlyStopping(patience=50),
                                  ],
                       epochs=self.epochs, verbose=2,
                       batch_size=128,
                       shuffle=True,
                       )
        """
        
        #self.load_model()

        return self

    def build_triplets(self, df_, y):
        anchor = []
        same = []
        diff = []
        labels = []

        for i in range(df_.shape[0]):
            li = y[i]
            
            df_same = df_[y == li]
            df_diff = df_[y != li]

            n_same = df_same.shape[0]
            n_diff = df_diff.shape[0]
            
            df_same = np.repeat(df_same, n_diff, axis=0)
            df_diff = np.repeat(df_diff, n_same, axis=0)

            index_ = np.arange(df_same.shape[0])
            np.random.shuffle(index_)

            index_ = index_[: int(0.05 * len(index_))]
            
            zip_same_diff = np.asarray(list(zip(df_same, df_diff)))
            for s, d in zip_same_diff[index_]:
                anchor.append(df_[i])
                same.append(s)
                diff.append(d)
                labels.append(li)

        index_ = np.arange(len(anchor))
        np.random.shuffle(index_)

        anchor = np.asarray(anchor)[index_]
        same = np.asarray(same)[index_]
        diff = np.asarray(diff)[index_]
        labels = np.asarray(labels)[index_]

        return anchor, same, diff, labels

    def load_model(self):
        if not hasattr(self, 'model'):
            self.model = self.build_model()

        try:
            self.model.load_weights(self.path)
        except:
            print('Error loading model')

    def build_model(self):
        def build_stream(monotonic):
            nfts = self.num_monotonic if monotonic else self.num_unconstrained
            input_ = KL.Input((self.num_features,))
            n = self.num_unconstrained
            
            if monotonic:
                last_ = KL.Lambda(lambda x: x[:, n:],)(input_)
            else:
                last_ = KL.Lambda(lambda x: x[:, : n],)(input_)

            if nfts > 0:
                constraint = non_neg() if monotonic else None

                num_dense = self.dense_monotonic_num if monotonic else \
                    self.dense_unconstrained_num

                for d in range(num_dense):
                    last_ = KL.Dense(self.dense_width,
                                     activation=self.dense_activation,
                                     use_bias=True,
                                     kernel_constraint=constraint,
                                     #kernel_regularizer=l2(self.l2),
                                     )(last_)

                if self.dropout is not None and self.dropout > 0.:
                    last_ = KL.Dropout(self.dropout)(last_)

            submodel = Model([input_], [last_])
            #submodel.summary()

            return submodel

        def build_submodel():
            # Global input
            input_ = KL.Input((self.num_features,))
            
            # Monotonic model
            monotonic_model = build_stream(True)(input_)
            unconstrained_model = build_stream(False)(input_)

            last_ = KL.Concatenate()([monotonic_model, unconstrained_model])

            for _ in range(self.dense_joint_num):
                last_ = KL.Dense(self.dense_width,
                                 activation=self.dense_activation,
                                 kernel_constraint=non_neg(),
                                 #kernel_regularizer=l2(self.l2),
                                 )(last_)

                if self.dropout is not None and self.dropout > 0.:
                    last_ = KL.Dropout(self.dropout)(last_)

            return Model(inputs=[input_], outputs=[last_])

        model = build_submodel()

        input_anchor = KL.Input((self.num_features,), name='anchor')
        input_pos = KL.Input((self.num_features,), name='positive')
        input_neg = KL.Input((self.num_features,), name='negative')

        anchor_embedding = model(input_anchor)
        pos_embedding = model(input_pos)
        neg_embedding = model(input_neg)

        euclidean_pos = KL.Lambda(lambda x: K.mean(K.square(x[0] - x[1]),
                                                   axis=1, keepdims=True),
                                  name='distance')
        euclidean_neg = KL.Lambda(lambda x: K.mean(K.square(x[0] - x[1]),
                                                   axis=1, keepdims=True))

        anchor_pos = euclidean_pos([anchor_embedding, pos_embedding])
        anchor_neg = euclidean_neg([anchor_embedding, neg_embedding])

        output_diff = KL.Lambda(lambda x: x[0] - x[1],
                                name='output-diff')([anchor_pos, anchor_neg])

        output_prob = KL.Dense(1, activation='sigmoid',
                               kernel_constraint=non_neg(),
                               name='output-prob')(anchor_embedding)

        model = Model(inputs=[input_anchor, input_pos, input_neg],
                      outputs=[output_prob, output_diff])

        self.embedding_model = Model([input_anchor], [anchor_embedding])
        self.distance_model = Model([input_anchor, input_pos],
                                    [anchor_pos])
        self.classification_model = Model([input_anchor], [output_prob])

        self.classification_model.compile('adadelta',
                                          'binary_crossentropy',
                                          metrics=['accuracy'])

        model.compile('adadelta',
                      loss={'output-diff': triplet_loss,
                            'output-prob': 'binary_crossentropy'},
                      loss_weights={'output-diff': 10.,
                                    'output-prob': 1},
                      metrics={'output-diff': triplet_correctness,
                               'output-prob': 'accuracy'},
                      )
        #model.summary()

        return model

    def predict_proba(self, df):
        ret = self.classification_model.predict(df[self.all_features])
        return np.hstack((1 - ret, ret))

    def predict(self, df):
        preds = self.classification_model.predict(df[self.all_features])
        preds = preds.ravel()
        #print(preds)
        return preds.round().astype(np.int)

    def explain(self, df):
        prob = self.explain_prob(df)
        neighbor_similar = self.explain_neighbor(df, similar=True)
        neighbor_opponent = self.explain_neighbor(df, similar=False)
        dec_rules_boundary = self.explain_impact(df, 'boundary')
        dec_rules_opponent = self.explain_impact(df, 'opponent')

        ret = list(zip(prob,
                       neighbor_similar,
                       neighbor_opponent,
                       dec_rules_boundary,
                       dec_rules_opponent,
                       ))
        return ret

    def explain_prob(self, df):
        probs = self.predict_proba(df)[:, 1]
        ret = []

        f = open(self.path, 'rb')
        serialized_model = f.read()
        f.close()

        for i, p in enumerate(probs):
            next_ = {'type': 'probability', 'value': p}

            if self.include_coverage:
                if p >= 0.5:
                    mask = probs >= p
                else:
                    mask = probs <= p

                next_['coverage'] = np.arange(df.shape[0])[mask]

                sprob = float_to_str(p)
                sexpl = (sprob.encode() + serialized_model)
                expl_length = min(len(sexpl),
                                  len(zlib.compress(sexpl, 9)))

                next_['compactness'] = len(serialized_model)

            ret.append(next_)
        return ret
 
    def explain_neighbor(self, df, similar=True):
        probs = self.predict_proba(df)[:, 1]
        classes = probs.round()

        kprobs = self.predict_proba(self.knowledge_df)[:, 1]
        kclasses = kprobs.round()
        
        dfm = df[self.all_features].as_matrix().astype(np.float)
        kdfm = \
            self.knowledge_df[self.all_features].as_matrix().astype(np.float)

        # Training set x Training set - point 2 point distance
        index_this = np.repeat(np.arange(kdfm.shape[0]), kdfm.shape[0])
        index_other = np.tile(np.arange(kdfm.shape[0]), kdfm.shape[0])
        print(index_this.shape, index_other.shape)
        # FIXME use batches here
        kp2p_dist = [self.distance_model.predict([kdfm[index_this[i: i+1]],
                                                  kdfm[index_other[i: i+1]]]) for i in range(index_this.shape[0])]
        kp2p_dist = np.asarray(kp2p_dist)
        print(kp2p_dist.shape)
        kp2p_dist = kp2p_dist.ravel()
        kp2p_dist = kp2p_dist.reshape((kdfm.shape[0], kdfm.shape[0]))
        
        # Test set x Training set - point 2 point distance
        index_this = np.repeat(np.arange(dfm.shape[0]), kdfm.shape[0])
        index_other = np.tile(np.arange(kdfm.shape[0]), dfm.shape[0])

        p2p_dist = self.distance_model.predict([dfm[index_this],
                                                kdfm[index_other]])
        p2p_dist = p2p_dist.ravel()
        p2p_dist = p2p_dist.reshape((dfm.shape[0], kdfm.shape[0]))

        p2p = [sorted(zip(classes[i] != kclasses if similar else
                          classes[i] == kclasses, d,
                          kclasses,
                          np.arange(len(d))))
               for i, d in enumerate(p2p_dist)]

        other = np.asarray([kdfm[x[0][-1]] for x in p2p])

        loss = self.distance_model.get_layer('distance').output

        grads = K.gradients(loss,
                            self.distance_model.get_layer('anchor').input)

        grads_fn = K.function([self.distance_model.get_layer('anchor').input,
                               self.distance_model.get_layer('positive').input,
                               ], grads)

        why_distance = grads_fn([dfm, other])[0]
        why_distance = np.abs(why_distance)
        why_distance = why_distance / np.maximum(1e-15,
                                                 np.sum(why_distance, axis=1,
                                                        keepdims=True))

        ret = []

        for i, (ordered_neighbors, why) in enumerate(zip(p2p, why_distance)):
            coverage = []
            
            if similar:
                # Exhaustive search of feature impact
                exp_details = []
                impact_sum = 0.

                orig_other = kdfm[ordered_neighbors[0][-1]]
                for ft in range(self.num_features):
                    other = orig_other.copy()
                    
                    values = list(set(kdfm[:, ft]))
                    others = np.repeat([other], len(values), axis=0)
                    others[:, ft] = values
                    
                    ndist = self.distance_model.predict([np.asarray([dfm[i]]),
                                                         others])
                    odist = ordered_neighbors[0][1]
                    ft_impact = np.max(np.abs(ndist - odist))

                    details = (ft_impact,
                               self.all_features[ft],
                               dfm[i][ft], orig_other[ft])
                    exp_details.append(details)
                    impact_sum += ft_impact

                if impact_sum == 0:
                    impact_sum = 1e-15

                exp_details = [(d / impact_sum, ft, vt, vo)
                               for d, ft, vt, vo in exp_details]
                exp_details = sorted(exp_details, key=lambda x: -x[0])

                cumsum = np.cumsum([x[0] for x in exp_details])

                aux = []
                for e, c in zip(exp_details, cumsum):
                    aux.append(e)
                    if c  >= self.explanation_completeness:
                        break
                exp_details = aux

                if self.include_coverage:
                    closest_dist = ordered_neighbors[0][2]
                    closest_neighbor = ordered_neighbors[0][-1]
                    mask = kp2p_dist[closest_neighbor] <= closest_dist
                    coverage = np.arange(kp2p_dist.shape[0])[mask]
            else:
                # Set values and check impact
                exp_details = []
                impact_sum = 0.

                orig_other = kdfm[ordered_neighbors[0][-1]]
                for ft in range(self.num_features):
                    other = orig_other.copy()
                    other[ft] = dfm[i][ft]

                    odist = ordered_neighbors[0][1]

                    ndist = self.distance_model.predict([np.asarray([dfm[i]]),
                                                         np.asarray([other])])
                    ndist = ndist.ravel()[0]
                    ft_impact = np.abs(ndist - odist)

                    details = (ft_impact,
                               self.all_features[ft],
                               dfm[i][ft], orig_other[ft])
                    exp_details.append(details)
                    impact_sum += ft_impact

                if impact_sum == 0:
                    impact_sum = 1e-15

                exp_details = [(d / impact_sum, ft, vt, vo)
                               for d, ft, vt, vo in exp_details]
                exp_details = sorted(exp_details, key=lambda x: -x[0])

                cumsum = np.cumsum([x[0] for x in exp_details])

                aux = []
                for e, c in zip(exp_details, cumsum):
                    aux.append(e)
                    if c  >= self.explanation_completeness:
                        break
                exp_details = aux
                
                if self.include_coverage:
                    closest_dist = ordered_neighbors[0][2]
                    closest_neighbor = ordered_neighbors[0][-1]

                    mask = kp2p_dist[closest_neighbor] >= closest_dist
                    coverage = np.arange(kp2p_dist.shape[0])[mask]

            explanation_type = 'similar' if similar else 'opponent'
            next_explanation = {'type': 'neighbor-%s' % explanation_type,
                                'neighbor': ordered_neighbors[0][-1],
                                'distance': ordered_neighbors[0][1],
                                'why': exp_details,
                                }

            if self.include_coverage:
                # Neighbor (total?)
                # Impact
                serialized_rule = ''
                nbits = math.ceil(np.log2(len(self.all_features)))

                for e in exp_details:
                    fid = self.all_features.index(e[1])
                    serialized_rule += ('%0' + str(nbits) + 'd') % fid
                    serialized_rule += float_to_str(e[0])
                    serialized_rule += float_to_str(e[-1])

                rule_length = min(len(serialized_rule),
                                len(zlib.compress(serialized_rule.encode(), 9)))

                next_explanation['compactness'] = rule_length
                next_explanation['coverage'] = coverage

            ret.append(next_explanation)

        return ret

    def explain_impact(self, df, impact_strategy):
        dfm = df[self.all_features].as_matrix().astype(np.float)
        kdfm = \
            self.knowledge_df[self.all_features].as_matrix().astype(np.float)

        probs = self.classification_model.predict(dfm).ravel()
        classes = probs.round().astype(int)

        prob_impact = np.zeros_like(dfm, dtype=np.float)
        overall_impact = np.zeros_like(dfm, dtype=np.float)
        current_value = np.zeros_like(dfm, dtype=np.float)
        local_optima = np.zeros_like(dfm, dtype=np.float)

        out_layer = self.model.get_layer('output-prob').output
        if impact_strategy == 'boundary':
            loss = mean_squared_error(K.constant([[0.5]]), out_layer)
        elif impact_strategy == 'opponent':
            loss = mean_squared_error(K.constant(1 - classes[:, np.newaxis]),
                                      out_layer)

        #loss = self.classification_model.get_layer('output-prob').output

        grads = K.gradients(loss, self.model.get_layer('anchor').input)
        grads_fn = K.function([self.model.get_layer('anchor').input], grads)

        for ft in range(dfm.shape[1]):
            ft_name = self.all_features[ft]

            dfm_iter = dfm.copy()
            
            dfm_optima = dfm.copy()

            alpha = 1.
            
            if ft_name in self.monotonic_features:
                min_value = kdfm[:, ft].min()
                max_value = kdfm[:, ft].max()
                dfm_iter[:, ft] = (classes == 0) * max_value + \
                    (classes == 1) * min_value
                dfm_optima[:, ft] = (classes == 0) * min_value + \
                    (classes == 1) * max_value
            else:
                learning_rate = np.sort(np.unique(kdfm[:, ft]))
                learning_rate = min(learning_rate[1:] -
                                    learning_rate[: -1]) / 2.

                for iter_ in range(100):
                    real_grads, = grads_fn([dfm_iter])
                    alpha = learning_rate / np.sqrt(iter_ + 1)

                    dfm_prev_iter = dfm_iter[:, ft].copy()
                    dfm_prev_optima = dfm_optima[:, ft].copy()
                    
                    dfm_iter[:, ft] = dfm_iter[:, ft] - alpha * \
                        real_grads[:, ft]

                    dfm_iter[:, ft] = np.maximum(kdfm[:, ft].min(),
                                                dfm_iter[:, ft])
                    dfm_iter[:, ft] = np.minimum(kdfm[:, ft].max(),
                                                dfm_iter[:, ft])

                    dfm_optima[:, ft] = dfm_iter[:, ft] + alpha * \
                        real_grads[:, ft]
                    dfm_optima[:, ft] = np.maximum(kdfm[:, ft].min(),
                                                   dfm_optima[:, ft])
                    dfm_optima[:, ft] = np.minimum(kdfm[:, ft].max(),
                                                   dfm_optima[:, ft])

                    tol_opponent = np.max(np.abs(dfm_prev_iter -
                                                 dfm_iter[:, ft]))
                    tol_optima = np.max(np.abs(dfm_prev_optima -
                                               dfm_optima[:, ft]))

                    if max(tol_opponent, tol_optima) < 1e-2:
                        break

            new_prob = self.classification_model.predict(dfm_iter).ravel()
            prob_impact[:, ft] = np.abs(new_prob - probs)
        
            for i in range(probs.shape[0]):
                thrs0 = dfm_iter[i, ft]
                thrs1 = dfm[i, ft]
                min_thrs = np.minimum(thrs0, thrs1)
                max_thrs = np.maximum(thrs0, thrs1)

                current_value[i, ft] = thrs1
                local_optima[i, ft] = dfm_optima[i, ft]

                lower_neighbors = (kdfm[:, ft] >= min_thrs) & \
                    (kdfm[:, ft] <= max_thrs)
                
                # FIXME: check if the normalization should be done considering
                # the local interval
                lower_neighbors = lower_neighbors.mean()
                ft_range = (dfm[i, ft] - dfm_iter[i, ft]) / \
                    (kdfm[:, ft].max() - kdfm[:, ft].min())

                overall_impact[i, ft] = (ft_range * prob_impact[i, ft])

                #overall_impact[i, ft] = math.sqrt(ft_range ** 2 +
                                                  #prob_impact[i, ft] ** 2)
                #overall_impact[i, ft] = (ft_range * #lower_neighbors *
                                         #prob_impact[i, ft])

        path = self.config['images']
        filenames = list(os.walk(path))[0][2]
        filenames = sorted(filenames)

        ret = []

        for i in range(dfm.shape[0]):
            impact = overall_impact[i]
            impact = impact / impact.sum()

            exp_details = sorted(zip(impact, self.all_features,
                                     current_value[i], local_optima[i],
                                     ), key=lambda x: -x[0])
            cumsum = np.cumsum([x[0] for x in exp_details])

            aux = []
            for e, c in zip(exp_details, cumsum):
                aux.append(e)
                if c  >= self.explanation_completeness:
                    break

            exp_details = aux
            exp_details = [{'impact': i, 'feature': ft, 'current-value': cv,
                            'end-point': ep}
                           for i, ft, cv, ep in exp_details]

            next_explanation = {'type': 'decision-rule-%s' % impact_strategy,
                                'rule': exp_details}
            
            if self.include_coverage:
                mask = np.ones(kdfm.shape[0], dtype=np.bool)
                for exp_branch in exp_details:
                    ft_index = self.all_features.index(exp_branch['feature'])
                    min_thrs = min(exp_branch['current-value'],
                                   exp_branch['end-point'])
                    max_thrs = max(exp_branch['current-value'],
                                   exp_branch['end-point'])

                    values = kdfm[:, ft_index]
                    mask = mask & (values >= min_thrs)
                    mask = mask & (values <= max_thrs)

                coverage = np.arange(kdfm.shape[0])[mask]
                next_explanation['coverage'] = coverage

            serialized_rule = ''
            nbits = math.ceil(np.log2(len(self.all_features)))

            for e in exp_details:
                fid = self.all_features.index(e['feature'])
                serialized_rule += ('%0' + str(nbits) + 'd') % fid
                serialized_rule += float_to_str(e['current-value'])
                if e['feature'] not in self.monotonic_features:
                    serialized_rule += float_to_str(e['end-point'])

            rule_length = min(len(serialized_rule),
                              len(zlib.compress(serialized_rule.encode(), 9)))
            next_explanation['compactness'] = rule_length

            ret.append(next_explanation)

        return ret
