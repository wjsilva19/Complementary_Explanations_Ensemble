from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import RobustScaler, Imputer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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
import base64
import pickle
import math
import zlib
import cv2
import os


np.random.seed(42)
tf.set_random_seed(42)

def float_to_str(i):
    ret = []
    value = np.asarray(abs(i), dtype=np.float32).view(np.int32).item()
    while value != 0:
        res = value % 256
        ret.append(chr(res))
        value = value // 256

    return ('0' if i < 0 else '1') + ''.join(ret)


class DecisionTreeExplanation(BaseEstimator, ClassifierMixin):
    def __init__(self, config, criterion='gini', max_depth=None,
                 include_coverage=True):
        self.config = config

        # Features
        self.all_features = config["unconstrained"] + config["monotonic"]

        # Tree
        self.criterion = criterion
        self.max_depth = max_depth
        
        self.include_coverage = include_coverage

    def fit(self, df, y):
        dfm = df[self.all_features].as_matrix()
        self.imputer = Imputer().fit(dfm)
        dfm = self.imputer.transform(dfm)

        self.dfm = dfm

        self.estimator = DecisionTreeClassifier(criterion=self.criterion,
                                                max_depth=self.max_depth)
        self.estimator.fit(dfm, y)

        return self

    def explain(self, df):
        dfm = df[self.all_features].as_matrix()
        dfm = self.imputer.transform(dfm)

        leave_id = self.estimator.apply(dfm)
        knowledge_leave_id = self.estimator.apply(self.dfm)

        # Now, it's possible to get the tests that were used to predict a
        # sample or a group of samples. First, let's make it for the sample.

        indicator = self.estimator.decision_path(dfm)
        n_nodes = self.estimator.tree_.node_count
        children_left = self.estimator.tree_.children_left
        children_right = self.estimator.tree_.children_right
        feature = self.estimator.tree_.feature

        threshold = self.estimator.tree_.threshold

        explanations = []
        for id_ in range(len(df)):
            node_index = indicator.indices[indicator.indptr[id_]:
                                           indicator.indptr[id_ + 1]]

            rule = []
            serialized_rule = ''
            nbits = math.ceil(np.log2(len(self.all_features)))

            for node_id in node_index:
                if leave_id[id_] == node_id:
                    break
                
                if dfm[id_, feature[node_id]] <= threshold[node_id]:
                    serialized_threshold_sign = 0
                    threshold_sign = "<="
                else:
                    serialized_threshold_sign = 1
                    threshold_sign = ">"
                
                rule.append((self.all_features[feature[node_id]],
                             threshold_sign,
                             threshold[node_id]))
                
                if len(serialized_rule) != 0:
                    serialized_rule += '|'
                
                serialized_rule += ('%0' + str(nbits) + 'd') % feature[node_id]
                serialized_rule += str(serialized_threshold_sign)
                serialized_rule += float_to_str(threshold[node_id])

            next_explanation = {'type': 'tree-rule',
                                'rule': rule}

            if self.include_coverage:
                index_ = np.arange(self.dfm.shape[0])
                index_ = index_[knowledge_leave_id == leave_id[id_]]
                next_explanation['coverage'] = index_

            rule_length = min(len(serialized_rule),
                              len(zlib.compress(serialized_rule.encode(), 9)))
            next_explanation['compactness'] = rule_length

            explanations.append([next_explanation])


        return explanations

    def predict(self, df):
        dfm = df[self.all_features].as_matrix()
        dfm = self.imputer.transform(dfm)

        return self.estimator.predict(dfm)

    def predict_proba(self, df):
        dfm = df[self.all_features].as_matrix()
        dfm = self.imputer.transform(dfm)

        return self.estimator.predict_proba(dfm)


class RandomForestExplanation(BaseEstimator, ClassifierMixin):
    def __init__(self, config, n_estimators=100, criterion='gini',
                 max_depth=None, include_coverage=True):
        self.config = config

        # Features
        self.all_features = config["unconstrained"] + config["monotonic"]

        # Tree
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        
        self.include_coverage = include_coverage

    def fit(self, df, y):
        dfm = df[self.all_features].as_matrix()

        self.estimator = RandomForestClassifier(n_estimator=self.n_estimators,
                                                criterion=self.criterion,
                                                max_depth=self.max_depth,
                                                random_state=42,
                                                oob_score=True)
        #self.estimator = DecisionTreeClassifier(criterion=self.criterion,
        #                                        max_depth=self.max_depth)
        self.estimator.fit(dfm, y)

        return self

    def explain(self, df):
        dfm = df[self.all_features].as_matrix()

        global_preds = self.predict(df)
        trees = self.estimator.estimators_
        explanations_per_tree = []

        for tree in trees:
            local_preds = tree.predict(dfm)
            index = np.arange(dfm.shape[0])[local_preds == global_preds]

            leave_id = tree.apply(dfm[index])

            # Now, it's possible to get the tests that were used to predict a
            # sample or a group of samples. First, let's make it for the sample.

            indicator = tree.decision_path(dfm[index])
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature

            threshold = tree.tree_.threshold

            explanations = [None for _ in range(len(dfm))]
            
            for id_ in range(len(dfm[index])):
                node_index = indicator.indices[indicator.indptr[id_]:
                                            indicator.indptr[id_ + 1]]

                rule = []
                serialized_rule = ''
                nbits = math.ceil(np.log2(len(self.all_features)))

                for node_id in node_index:
                    if leave_id[id_] == node_id:
                        break

                    if dfm[index][id_, feature[node_id]] <= threshold[node_id]:
                        serialized_threshold_sign = 0
                        threshold_sign = "<="
                    else:
                        serialized_threshold_sign = 1
                        threshold_sign = ">"

                    rule.append((self.all_features[feature[node_id]],
                                threshold_sign,
                                threshold[node_id]))

                    if len(serialized_rule) != 0:
                        serialized_rule += '|'
                    serialized_rule += ('%0' + str(nbits) + 'd') % feature[node_id]
                    serialized_rule += str(serialized_threshold_sign)
                    serialized_rule += float_to_str(threshold[node_id])

                next_explanation = {'type': 'tree-rule',
                                    'rule': rule}

                if self.include_coverage:
                    index_ = np.arange(dfm[index].shape[0])
                    index_ = index_[leave_id == leave_id[id_]]
                    next_explanation['coverage'] = index_

                rule_length = min(len(serialized_rule),
                                len(zlib.compress(serialized_rule.encode(), 9)))
                next_explanation['compactness'] = rule_length

                explanations[index[id_]] = next_explanation
            
            explanations_per_tree.append(explanations)

        return explanations

    def predict(self, df):
        dfm = df[self.all_features].as_matrix()
        return self.estimator.predict(dfm)

    def predict_proba(self, df):
        dfm = df[self.all_features].as_matrix()
        return self.estimator.predict_proba(dfm)


class NeighborExplanation(BaseEstimator, ClassifierMixin):
    def __init__(self, config, neighbors=1, weights='uniform',
                 include_coverage=True):
        self.config = config
        self.neighbors = neighbors
        self.weights = weights

        # Features
        self.all_features = config["unconstrained"] + config["monotonic"]

        self.include_coverage = include_coverage

    def fit(self, df, y):
        self.knowledge_df = df.copy()
        self.knowledge_y = y.copy()

        dfm = df[self.all_features].as_matrix()
        
        self.estimator = KNeighborsClassifier(n_neighbors=self.neighbors,
                                              weights=self.weights)
        self.estimator.fit(dfm, y)

        return self

    def explain(self, df):
        neighbor_similar = self.explain_neighbor(df, similar=True)
        neighbor_opponent = self.explain_neighbor(df, similar=False)

        ret = list(zip(neighbor_similar,
                       neighbor_opponent,
                       ))

        return ret

    def explain_neighbor(self, df, similar=True):
        k_classes = self.knowledge_y
        new_classes = self.predict(df)

        dfm = df[self.all_features].as_matrix().astype(np.float)

        kdfm = self.knowledge_df[self.all_features]
        kdfm = kdfm.as_matrix().astype(np.float)

        distances = euclidean_distances(dfm, kdfm)
        kdistances = euclidean_distances(kdfm, kdfm)
        
        explanations = []
        name = 'similar' if similar else 'opponent'

        for id_ in range(len(df)):
            dists = distances[id_]

            if similar:
                mask = k_classes == new_classes[id_]
            else:
                mask = k_classes != new_classes[id_]

            index = np.arange(dists.shape[0])[mask]
            neighbor = index[np.argmin(dists[mask])]
            d = distances[id_][neighbor]

            next_explanation = {'type': 'euclidean-neighbor-%s' % name,
                                'neighbor': neighbor,
                                'distance': d
                                }

            if self.include_coverage:
                cov = np.arange(kdfm.shape[0])
                if similar:
                    cov = cov[kdistances[neighbor] <= d]
                else:
                    cov = cov[kdistances[neighbor] >= d]

                next_explanation['coverage'] = cov

                serialized_rule = list(kdfm[neighbor]) + [d]
                serialized_rule = ''.join(map(float_to_str, serialized_rule))
                rule_length = min(len(serialized_rule),
                              len(zlib.compress(serialized_rule.encode(), 9)))

                next_explanation['compactness'] = rule_length

            explanations.append(next_explanation)

        return explanations

    def predict(self, df):
        dfm = df[self.all_features].as_matrix()
        return self.estimator.predict(dfm)

    def predict_proba(self, df):
        dfm = df[self.all_features].as_matrix()
        return self.estimator.predict_proba(dfm)


#class DeepInterpretableModel(BaseEstimator, ClassifierMixin):
    #def __init__(self, config,
                 #unconstrained_features=[], monotonic_features=[],
                 #dense_width=5, dense_stream_num=1, dense_joint_num=1,
                 #dense_activation='tanh',
                 #dropout=0., l2=1e-2,
                 #l1=1e-5, path='model.h5', epochs=500, batch_size=64):

        #self.config = config

        ## Features
        #self.unconstrained_features = unconstrained_features
        #self.monotonic_features = monotonic_features

        #self.all_features = unconstrained_features + monotonic_features
        #self.num_features = len(self.all_features)

        #self.num_monotonic = len(self.monotonic_features)
        #self.num_unconstrained = len(self.unconstrained_features)
        
        ## Architecture
        #self.dense_width = dense_width
        #self.dense_stream_num = dense_stream_num
        #self.dense_joint_num = dense_joint_num
        #self.dense_activation = dense_activation
        
        ## Regularization
        #self.l2 = l2
        #self.l1 = l1
        #self.dropout = dropout

        ## Utils
        #self.path = path
        #self.batch_size = batch_size
        #self.epochs = epochs

    #def fit(self, df, y):
        #self.knowledge_df = df.copy()
        #self.knowledge_y = y.copy()

        #self.load_model()

        #dfm = df[self.all_features].as_matrix()
        ##self.scaler = RobustScaler((10, 90)).fit(dfm)
        #df_e = dfm  # self.scaler.transform(dfm)
        
        #self.model.fit(df_e,
                       #y.astype(np.float),
                       #callbacks=[ModelCheckpoint(self.path,
                                                  #save_best_only=True,
                                                  #verbose=1),
                                  #EarlyStopping(patience=100),],
                       #epochs=self.epochs, verbose=2,
                       #batch_size=self.batch_size,
                       #shuffle=True, validation_split=0.3)

        #self.load_model()

        #return self

    #def load_model(self):
        #if not hasattr(self, 'model'):
            #self.model = self.build_model()

        #try:
            #self.model.load_weights(self.path)
        #except:
            #print('Error loading model')

    #def build_model(self):
        #def build_stream(monotonic):
            #nfts = self.num_monotonic if monotonic else self.num_unconstrained
            #input_ = KL.Input((self.num_features,))
            #n = self.num_unconstrained
            
            #if monotonic:
                #last_ = KL.Lambda(lambda x: x[:, n:],)(input_)
            #else:
                #last_ = KL.Lambda(lambda x: x[:, : n],)(input_)

            #if nfts > 0:
                #constraint = non_neg() if monotonic else None
                #for d in range(self.dense_stream_num):
                    #last_ = KL.Dense(self.dense_width,
                                     #activation=self.dense_activation,
                                     #use_bias=True,
                                     #kernel_constraint=constraint,
                                     #kernel_regularizer=l2(self.l2),
                                     ##activity_regularizer=l1(self.l1),
                                     #)(last_)

                #if self.dropout is not None:
                    #last_ = KL.Dropout(self.dropout)(last_)

            #submodel = Model([input_], [last_])
            #submodel.summary()
            #return submodel

        ## Global input
        #input_ = KL.Input((self.num_features,), name='input')
        
        ## Monotonic model
        #monotonic_model = build_stream(True)(input_)
        #unconstrained_model = build_stream(False)(input_)

        #last_ = KL.Concatenate()([monotonic_model, unconstrained_model])

        #for _ in range(self.dense_joint_num):
            #last_ = KL.Dense(self.dense_width,
                             #activation=self.dense_activation,
                             #kernel_constraint=non_neg(),
                             #kernel_regularizer=l2(self.l2),
                             ##activity_regularizer=l1(self.l1),
                             #)(last_)
            #if self.dropout is not None:
                #last_ = KL.Dropout(self.dropout)(last_)

        #self.model_to_embedding = Model(inputs=[input_],
                                        #outputs=[last_])

        #constraint = non_neg() if self.num_monotonic > 0 else None
        #output = KL.Dense(1, activation='sigmoid', name='output',
                          #kernel_constraint=constraint,
                          ##kernel_regularizer=l2(self.l2),
                          #)(last_)

        #model = Model(inputs=[input_],
                      #outputs=[output])

        #model.compile('adam', loss='binary_crossentropy',
                      #metrics=['accuracy'])
        #model.summary()
        ##os.sys.exit(0)
        #return model

    #def predict_proba(self, df):
        #ret = self.model.predict(df[self.all_features])
        #return np.hstack((1 - ret, ret))

    #def predict(self, df):
        #return self.predict_proba(df)[:, 1].round()

    #def explain(self, df):
        #raise NotImplementedError



#class DeepLocalExplanation(DeepInterpretableModel):
    #def __init__(self,
                 #config, unconstrained_features=[], monotonic_features=[],
                 #dense_width=5, dense_stream_num=1, dense_joint_num=1,
                 #dense_activation='tanh',
                 #dropout=0.0, l2=1e-2,
                 #l1=1e-5, path='model.h5', epochs=500, batch_size=64):

        #super().__init__(config,
                         #unconstrained_features, monotonic_features,
                         #dense_width, dense_stream_num, dense_joint_num,
                         #dense_activation, dropout, l2, l1, path, epochs,
                         #batch_size)

    #def explain(self, df, y):
        #dfm = df[self.all_features].as_matrix().astype(np.float)

        #probs = self.model.predict(dfm).ravel()
        #classes = probs.round().astype(int)

        #prob_impact = np.zeros_like(dfm, dtype=np.float)
        #overall_impact = np.zeros_like(dfm, dtype=np.float)
        #current_value = np.zeros_like(dfm, dtype=np.float)
        #local_optima = np.zeros_like(dfm, dtype=np.float)

        #loss = mean_squared_error(#K.constant([[0.]]),  # 1 - classes[:, np.newaxis]),
                                  ##K.constant(1 - classes[:, np.newaxis]),
                                  #K.constant([[0.5]]),
                                  #self.model.get_layer('output').output)
        #loss = self.model.get_layer('output').output

        #grads = K.gradients(loss, self.model.get_layer('input').input)
        #grads_fn = K.function([self.model.get_layer('input').input], grads)

        #for ft in range(dfm.shape[1]):
            #ft_name = self.all_features[ft]
            ##if ft_name in self.monotonic_features:
                ##continue
            
            #print(ft_name, ft_name in self.monotonic_features)

            #dfm_iter = dfm.copy()
            #dfm_optima = dfm.copy()

            #alpha = 1.
            
            #if ft_name in self.monotonic_features:
                #min_value = dfm[:, ft].min()
                #max_value = dfm[:, ft].max()
                #dfm_iter[:, ft] = (classes == 0) * max_value + \
                    #(classes == 1) * min_value
                #dfm_optima[:, ft] = (classes == 0) * min_value + \
                    #(classes == 1) * max_value
            #else:
                #learning_rate = np.sort(np.unique(dfm[:, ft]))
                #learning_rate = min(learning_rate[1:] -
                                    #learning_rate[: -1]) / 2.
                
                #for iter_ in range(100):
                    #real_grads, = np.sign(grads_fn([dfm_iter]))
                    #alpha = learning_rate

                    #dfm_iter[:, ft] = dfm_iter[:, ft] - alpha * \
                        #real_grads[:, ft]

                    #dfm_iter[:, ft] = np.maximum(dfm[:, ft].min(),
                                                #dfm_iter[:, ft])
                    #dfm_iter[:, ft] = np.minimum(dfm[:, ft].max(),
                                                #dfm_iter[:, ft])
                    
                    #dfm_optima[:, ft] = dfm_iter[:, ft] + alpha * \
                        #real_grads[:, ft]
                    #dfm_optima[:, ft] = np.maximum(dfm[:, ft].min(),
                                                   #dfm_optima[:, ft])
                    #dfm_optima[:, ft] = np.minimum(dfm[:, ft].max(),
                                                   #dfm_optima[:, ft])

            #"""
            #for pr, init_v, curr_v in zip(probs, dfm[:, ft], dfm_iter[:, ft]):
                #print('%.4f %.4f %.4f %.4f %.4f' % (pr, dfm[:, ft].min(),
                                                    #init_v, curr_v,
                                                    #dfm[:, ft].max()))
            #"""
            #new_prob = self.model.predict(dfm_iter).ravel()
            #prob_impact[:, ft] = np.abs(new_prob - probs)
        
            #for i in range(probs.shape[0]):
                #thrs0 = dfm_iter[i, ft]
                #thrs1 = dfm[i, ft]
                #min_thrs = np.minimum(thrs0, thrs1)
                #max_thrs = np.maximum(thrs0, thrs1)

                #current_value[i, ft] = thrs1
                #local_optima[i, ft] = dfm_optima[i, ft]

                #lower_neighbors = (dfm[:, ft] >= min_thrs) & \
                    #(dfm[:, ft] <= max_thrs)
                
                ## FIXME: check if the normalization should be done considering
                ## the local interval
                #lower_neighbors = lower_neighbors.mean()

                #overall_impact[i, ft] = lower_neighbors * prob_impact[i, ft]

        #path = self.config['images']
        #filenames = list(os.walk(path))[0][2]
        #filenames = sorted(filenames)

        #for i in range(dfm.shape[0]):
            #impact = overall_impact[i]
            #impact = impact / impact.sum()
            #print(filenames[i])
            #print(y[i], probs[i])
            #print(sorted(zip(impact, self.all_features,
                             #current_value[i], local_optima[i],
                             #))[::-1][:7])
            #print()
            ##os.sys.exit(0)
            #plt.imshow(cv2.imread(os.path.join(path, filenames[i]))[:, :, ::-1])
            #plt.show()
        ##plt.show()
        
        #for i in range(dfm.shape[0]):
            #for j in range(self.num_features):
                #if self.all_features[j] in self.monotonic_features:
                    #if classes[i] == 0:
                        #optimal_value = dfm[:, j].max()
                    #else:
                        #optimal_value = dfm[:, j].min()
                #else:
                    #continue
                    
                    #os.sys.exit(0)

                #next_vector = dfm[i].copy()
                #next_vector[j] = optimal_value

                #new_prob = self.model.predict(np.asarray([next_vector]))
                #next_prob_impact = np.abs(new_prob.ravel()[0] - probs[i])
                
                #prob_impact[i][j] = next_prob_impact

        #os.sys.exit(0)

        ## TODO
        #"""
        #THIS IS THE ONE THAT WORKS!!!!!
        #print(self.all_features)
        #dfm = df[self.all_features].as_matrix().astype(np.float)


        #filenames = list(os.walk(self.config['images']))[0][2]
        #filenames = sorted(filenames)

        #for i in range(dfm.shape[0]):
            #lower_neighbors = (dfm <= dfm[i, :]).mean(axis=0)
            #impact = lower_neighbors * prob_impact[i]
            #impact = impact / impact.sum()
            #print(filenames[i])
            #print(y[i], probs[i])
            #print(sorted(zip(impact, self.all_features))[::-1][:3])
            #print()
            ##os.sys.exit(0)
            #plt.imshow(cv2.imread(os.path.join(self.config['images'], filenames[i]))[:, :, ::-1])
            #plt.show()
        ##plt.show()
    #"""


## Same class, Opposite class, Uncertainty
## Magnitude vs. rarity

#class GradientExplanation(DeepInterpretableModel):
    #def __init__(self,
                 #config, loss='best', strategy='rarity', explanation_size=1,
                 #unconstrained_features=[], monotonic_features=[],
                 #dense_width=5, dense_stream_num=1, dense_joint_num=1,
                 #dense_activation='tanh',
                 #dropout=0.,
                 #l1=1e-5, path='model.h5', epochs=500, batch_size=64):

        #super().__init__(config, unconstrained_features, monotonic_features,
                         #dense_width, dense_stream_num, dense_joint_num,
                         #dense_activation, dropout, l1, path, epochs,
                         #batch_size)

        #self.loss = loss
        #self.strategy = strategy
        #self.explanation_size = explanation_size

    #def explain(self, df, y):
        ## TODO
        #dfm = df[self.all_features].as_matrix().astype(np.float)
        #df_e = dfm  # self.scaler.transform(dfm)
        
        #probs = self.predict_proba(df)[:, 1]

        ## TODO: check if 0.5 or random decision for uncertainty/opposite

        #if self.loss == 'best':
            #loss = mean_squared_error(K.constant([[0.]]),
                                      #self.model.get_layer('output').output)
        #elif self.loss == 'uncertainty':
            #loss = mean_squared_error(K.constant([[0.5]]),
                                      #self.model.get_layer('output').output)
        #elif self.loss == 'opposite':
            #opposite = 1 - probs.round()
            #loss = mean_squared_error(K.constant(opposite),
                                      #self.model.get_layer('output').output)
        #elif self.loss == 'same':
            #same = probs.round()
            #loss = mean_squared_error(K.constant(same),
                                      #self.model.get_layer('output').output)
        #else:
            #raise NotImplementedError

        #grads = K.gradients(loss, self.model.get_layer('input').input)
        #grads_fn = K.function([self.model.get_layer('input').input], grads)

        #real_grads, = grads_fn([df_e])
        ##real_grads /= np.max(np.abs(real_grads))

        ##goi = real_grads[:, 0]
        ##goi = np.log(goi - goi.min() + 1e-15)
        ##goi = goi[(goi > np.percentile(goi, 5)) & (goi < np.percentile(goi, 95))]
        ##plt.hist(goi, 30)
        ##plt.show()

        #filenames = list(os.walk(self.config['images']))[0][2]
        #filenames = sorted(filenames)

        #if self.strategy == 'magnitude':
            ## FIXME
            
            #abs_grad = np.abs(real_grads)
            #abs_grad = abs_grad * probs[:, np.newaxis] # * df_e
            #abs_grad = [sorted(zip(-np.abs(g), self.all_features, g
                                   ##np.arange(real_grads.shape[1])
                                   #))
                        #for g in abs_grad]

            #sorted_imgs = []
            #for i, e in enumerate(abs_grad):
                ##_, e, _ = zip(*e)
                #print(filenames[i])
                #print(i, probs[i], y[i])
                #print(list(zip(*e))[1])
                #print()
                #sorted_imgs.append((e[0][0], i))
                
                ##img = cv2.imread(os.path.join(self.config['images'], filenames[i]))
                ##plt.imshow(img[:, :, ::-1])
                ##plt.show()

            #os.sys.exit(0)

            ## color 11
            ## scar 18
            ## asym 1
            #print(self.all_features)
            #sorted_grads = np.abs(real_grads)[:, 3]
            #sorted_grads = sorted(list(enumerate(sorted_grads)),
                                  #key=lambda x: -x[1])

            

            #for i, g in sorted_grads:#[::-1]:
                #print(filenames[i], '%.7f' % g)
                #img = cv2.imread(os.path.join(self.config['images'], filenames[i]))
                #plt.imshow(img[:, :, ::-1])
                #plt.show()
            #print(sorted_imgs)
            #raise NotImplementedError
        #elif self.strategy == 'rarity':
            #print(list(real_grads[:, 0]))
            #print(sorted(list(real_grads[:, 0])))
            #print()
            
            #abs_grad = np.abs(real_grads)
            #abs_grad = abs_grad * probs[:, np.newaxis] # * df_e
            
            ##abs_grad = RobustScaler(quantile_range=(10, 90)).fit_transform(abs_grad)
            ##abs_grad = (abs_grad - abs_grad.mean(axis=0)) / \
            ##    abs_grad.std(axis=0)

            #abs_grad = abs_grad.T
            #abs_grad = np.asarray([rankdata(d) for d in abs_grad]).T
            ##abs_grad = abs_grad / abs_grad.shape[0]

            #grad_rank = [sorted([(-g, self.all_features[i])
                                 #for i, g in enumerate(obs)])
                         #for obs in abs_grad]

            #for i, g in enumerate(grad_rank):
                #print(i, y[i])
                #print([x for x in g if x[0] == g[0][0]])
                #img = cv2.imread(os.path.join(self.config['images'],
                                              #filenames[i]))
                #plt.imshow(img[:, :, ::-1])
                #plt.show()
                ##gsum = [(g[1][0], g[0]) for g in g]
                ##gsum = {k: np.sum(np.abs([s[1] for s in gsum if s[0] == k]))
                        ##for k in ['p', 's', 'c']}
                ##print(sorted(gsum.items(), key=lambda x: -x[1]))
                #print()

            ##plt.subplot(121)
            ##plt.imshow(np.abs(real_grads) / np.max(np.abs(real_grads)))
            ##plt.subplot(122)
            ##plt.imshow(abs_grad)
            ##plt.show()
            
            ## FIXME
            #raise NotImplementedError
        #else:
            #raise NotImplementedError


#class GradientDescentExplanation(DeepInterpretableModel):
    #def __init__(self,
                 #config, loss='best', strategy='magnitude', explanation_size=1,
                 #unconstrained_features=[], monotonic_features=[],
                 #dense_width=5, dense_stream_num=1, dense_joint_num=1,
                 #dense_activation='tanh',
                 #dropout=0.,
                 #l1=1e-5, path='model.h5', epochs=500, batch_size=64):

        #super().__init__(config, unconstrained_features, monotonic_features,
                         #dense_width, dense_stream_num, dense_joint_num,
                         #dense_activation, dropout, l1, path, epochs,
                         #batch_size)

        #self.loss = loss
        #self.strategy = strategy
        #self.explanation_size = explanation_size

    #def explain(self, df, y):
        ## TODO
        #probs = self.predict_proba(df)[:, 1]

        ## TODO: check if 0.5 or random decision for uncertainty/opposite

        #if self.loss == 'best':
            #loss = mean_squared_error(K.constant([[0.]]),
                                      #self.model.get_layer('output').output)
        #elif self.loss == 'uncertainty':
            #loss = mean_squared_error(K.constant([[0.5]]),
                                      #self.model.get_layer('output').output)
        #elif self.loss == 'opposite':
            #opposite = 1 - probs.round()
            #loss = mean_squared_error(K.constant(opposite),
                                      #self.model.get_layer('output').output)
        #elif self.loss == 'same':
            #same = probs.round()
            #loss = mean_squared_error(K.constant(same),
                                      #self.model.get_layer('output').output)
        #else:
            #raise NotImplementedError

        #grads = K.gradients(loss, self.model.get_layer('input').input)
        #grads_fn = K.function([self.model.get_layer('input').input], grads)
        #loss_fn = loss
        
        #dfm = df[self.all_features].as_matrix().astype(np.float)
        #dfm_init = dfm.copy()
        
        #orig_probs = self.model.predict(dfm).ravel()
        
        #alpha = 1e-2
        #for iter_ in range(100):
            #real_grads, = grads_fn([dfm])
            ##real_grads /= np.abs(real_grads).max()
            #dfm = dfm - alpha * real_grads
        
        #dfm_end = dfm.copy()
        #new_probs = self.model.predict(dfm_end).ravel()

        #dfm_diff = np.abs(dfm_init - dfm_end)
        ##print(self.model.predict(dfm).ravel())
        
        #filenames = list(os.walk(self.config['images']))[0][2]
        #filenames = sorted(filenames)
        #for i in range(len(dfm_diff)):
            #next_diff = dfm_diff[i]
            #prob_diff = np.abs(new_probs[i] - orig_probs[i])
            #next_diff = prob_diff * next_diff
            
            ##next_diff /= np.sum(np.abs(next_diff))

            #sorted_diff = sorted(zip(next_diff, self.all_features))[::-1]
            #print(sorted_diff)
            #print()
            #print()
            #img = cv2.imread(os.path.join(self.config['images'], filenames[i]))
            #plt.imshow(img[:, :, ::-1])
            #plt.show()
        #print(sorted_imgs)
        
        #print(real_grads)



#class NeighborExplanation(DeepInterpretableModel):
    #def explain(self, df, y):
        ## TODO
        ##probs = self.predict_proba(df)
        #dfm = df[self.all_features].as_matrix().astype(np.float)
        #embeddings = self.model_to_embedding.predict(dfm)
        
        #from sklearn.manifold import TSNE
        #from sklearn.metrics.pairwise import pairwise_distances
        
        #filenames = list(os.walk(self.config['images']))[0][2]
        #filenames = sorted(filenames)

        #p2p = pairwise_distances(embeddings)
        #p2p = [(i, y[i], sorted(zip(d, np.arange(len(d)), y))[1]) for i, d in enumerate(p2p)]
        
        #other = np.asarray([dfm[x[2][1]] for x in p2p])
        
        ##self.model_to_distance.predict(dfm, dfm[p2p
        ##Model([input_anchor, input_pos],
                                       ##[anchor_pos])
        ##loss = 1. / (1 + self.model_to_distance.get_layer('distance').output)
        #loss = self.model_to_distance.get_layer('distance').output

        #grads = K.gradients(loss,
                            #self.model_to_distance.get_layer('anchor').input)
        
        #grads_fn = K.function([self.model_to_distance.get_layer('anchor').input,
                               #self.model_to_distance.get_layer('positive').input,
                               ##self.model.get_layer('negative').input,
                               #], grads)

        #noise = np.random.normal(loc=0, scale=1e-5, size=dfm.shape)
        #why_similar = grads_fn([dfm, other #+ noise
                                #])[0]

        #for i, li, (d, io, lo) in p2p:
            #print(i, io, li, lo)

            #if max(li, lo) == 0:
                #continue

            #if np.all(dfm[i] == other[i]):
                #continue

            #to_print = [(self.all_features[j], dfm[i][j], other[i][j])
                        #for j in range(len(self.all_features))]
            #to_print = sorted(to_print)
            #for tp in to_print:
                #print(' '.join(map(str,tp)))
        
            #print(sorted(zip(np.abs(why_similar[i]), self.all_features))[::-1])
            
            #plt.subplot(121)
            #img = cv2.imread(os.path.join(self.config['images'], filenames[i]))
            #plt.imshow(img[:, :, ::-1])
            #plt.subplot(122)
            #img = cv2.imread(os.path.join(self.config['images'], filenames[io]))
            #plt.imshow(img[:, :, ::-1])
            #plt.show()
        #"""
        #"""
        #print(p2p)
        ##plt.imshow(p2p)
        ##plt.show()

        #proj = TSNE().fit_transform(embeddings)
        #plt.scatter(proj[:, 0], proj[:, 1], c=y)#probs[:, 1])
        #plt.show()

        #print(embeddings.shape)
        ##return np.asarray([{'probability': p for p in probs}])


#def triplet_loss(_, y_pred):
    #alpha = 0  # 0.2
    #return K.sum(K.maximum(0., y_pred + alpha ))

#def triplet_correctness(_, y_pred):
    #return K.mean(y_pred <= 0)


#class NeighborTripletLossExplanation(NeighborExplanation):
    #def fit(self, df, y):
        #self.knowledge_df = df.copy()
        #self.knowledge_y = y.copy()

        #self.load_model()

        ## Create tuples
        #df_ = df[self.all_features].as_matrix().astype(np.float)
        
        #anchor = []
        #same = []
        #diff = []

        #for i in range(df_.shape[0]):
            #li = y[i]
            
            #df_same = df_[y == li]
            #df_diff = df_[y != li]

            #n_same = df_same.shape[0]
            #n_diff = df_diff.shape[0]
            
            #df_same = np.repeat(df_same, n_diff, axis=0)
            #df_diff = np.repeat(df_diff, n_same, axis=0)
            #index_ = np.arange(df_same.shape[0])
            #np.random.shuffle(index_)
            
            #zip_same_diff = np.asarray(list(zip(df_same, df_diff)))
            #for s, d in zip_same_diff[index_[: int(0.2 * len(index_))]]:
                #anchor.append(df_[i])
                #same.append(s)
                #diff.append(d)
            
        #anchor = np.asarray(anchor)
        #same = np.asarray(same)
        #diff = np.asarray(diff)

        #self.model.fit({'anchor': anchor,
                        #'positive': same,
                        #'negative': diff},
                       #np.zeros((anchor.shape[0], 1)),
                       #callbacks=[ModelCheckpoint(self.path,
                                                  #save_best_only=True,
                                                  #verbose=1),
                                  #EarlyStopping(patience=100),],
                       #epochs=self.epochs, verbose=2,
                       #batch_size=512, #self.batch_size,
                       #shuffle=True,
                       #validation_split=0.3
                       #)

        #self.load_model()

        #return self

    #def load_model(self):
        #if not hasattr(self, 'model'):
            #self.model = self.build_model()

        #try:
            #self.model.load_weights(self.path)
        #except:
            #print('Error loading model')

    #def build_model(self):
        #def build_stream(monotonic):
            #nfts = self.num_monotonic if monotonic else self.num_unconstrained
            #input_ = KL.Input((self.num_features,))
            #n = self.num_unconstrained
            
            #if monotonic:
                #last_ = KL.Lambda(lambda x: x[:, n:],)(input_)
            #else:
                #last_ = KL.Lambda(lambda x: x[:, : n],)(input_)

            #if nfts > 0:
                #constraint = non_neg() if monotonic else None
                #for d in range(self.dense_stream_num):
                    #last_ = KL.Dense(self.dense_width,
                                     #activation=self.dense_activation,
                                     #use_bias=True,
                                     #kernel_constraint=constraint,
                                     ##activity_regularizer=l1(self.l1),
                                     #)(last_)

                #if self.dropout is not None:
                    #last_ = KL.Dropout(self.dropout)(last_)

            #submodel = Model([input_], [last_])
            #submodel.summary()
            #return submodel

        #def build_submodel():
            ## Global input
            #input_ = KL.Input((self.num_features,))
            
            ## Monotonic model
            #monotonic_model = build_stream(True)(input_)
            #unconstrained_model = build_stream(False)(input_)

            #last_ = KL.Concatenate()([monotonic_model, unconstrained_model])

            #for _ in range(self.dense_joint_num):
                #last_ = KL.Dense(self.dense_width,
                                #activation=self.dense_activation,
                                #kernel_constraint=non_neg(),
                                ##activity_regularizer=l1(self.l1),
                                #)(last_)
                #if self.dropout is not None:
                    #last_ = KL.Dropout(self.dropout)(last_)

            #return Model(inputs=[input_], outputs=[last_])

        #model = build_submodel()

        #input_anchor = KL.Input((self.num_features,), name='anchor')
        #input_pos = KL.Input((self.num_features,), name='positive')
        #input_neg = KL.Input((self.num_features,), name='negative')
        
        #anchor_embedding = model(input_anchor)
        #pos_embedding = model(input_pos)
        #neg_embedding = model(input_neg)

        #self.model_to_embedding = Model([input_anchor], [anchor_embedding])

        #euclidean_pos = KL.Lambda(lambda x: K.sum(K.square(x[0] - x[1]),
                                                  #axis=1, keepdims=True),
                                  #name='distance')
        #euclidean_neg = KL.Lambda(lambda x: K.sum(K.square(x[0] - x[1]),
                                                  #axis=1, keepdims=True))

        #anchor_pos = euclidean_pos([anchor_embedding, pos_embedding])
        #self.model_to_distance = Model([input_anchor, input_pos],
                                       #[anchor_pos])

        #anchor_neg = euclidean_neg([anchor_embedding, neg_embedding])

        #diff = KL.Lambda(lambda x: x[0] - x[1], name='output')([anchor_pos,
                                                                #anchor_neg])
        
        #model = Model(inputs=[input_anchor, input_pos, input_neg],
                      #outputs=[diff])

        #model.compile('adam', loss=triplet_loss,
                      #metrics=[triplet_correctness])
        #model.summary()

        ##os.sys.exit(0)
        #return model

    #def predict_proba(self, df):
        #ret = self.model.predict(df[self.all_features].astype(np.float))
        #return np.hstack((1 - ret, ret))

    #def predict(self, df):
        #return self.predict_proba(df)[:, 1].round()

    ##def explain(self, df, y):
        ##raise NotImplementedError
