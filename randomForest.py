# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 02:17:39 2019

@author: brayn

MODEL:Random Forest
"""

import Utils
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
"""
training_data = Utils.load_database('data/BASE-PREPROCESSED(TRAIN).gz')

training_data = training_data[:, :-1]

pagantes, inadiplentes  = Utils.separate_classes(training_data)

training_data = Utils.replicate_shuffle_merge(pagantes, inadiplentes)

X, Y = Utils.get_input_output(training_data)

X_train, y_train, X_test, y_test, X_val, y_val = Utils.separate_train_test_eval(X, Y)

print(X_train)

X_train, X_test, X_val = Utils.normalize(X_train, X_test, X_val)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
db_train = Utils.load_database('data/BASE-TRAIN.gz', sep='\t')
db_test = Utils.load_database('data/BASE-TEST.gz', sep='\t')
db_val = Utils.load_database('data/BASE-VALID.gz', sep='\t')
print(db_train[:6])
X_train, y_train, X_test, y_test, X_val, y_val = (db_train[:, 2:], db_train[:, 1],
                                                  db_test[:, 2:], db_test[:, 1],
                                                  db_val[:, 2:], db_val[:, 1])

db_train = Utils.load_database('data/TRAIN_PREPROC')
db_test = Utils.load_database('data/TEST_PREPROC')
db_val = Utils.load_database('data/VAL_PREPROC')

X_train, y_train, X_test, y_test, X_val, y_val = (db_train[:, :-1], db_train[:, -1],
                                                  db_test[:, :-1], db_test[:, -1],
                                                  db_val[:, :-1], db_val[:, -1])
"""
training_data = Utils.load_database('data/BASE-PREPROCESSED(TRAIN).gz', sep = '\t' )
training_data = training_data[:, :-1]
X, Y = Utils.get_input_output(training_data)
X_train, y_train, X_test, y_test, X_val, y_val = Utils.separate_train_test_eval(X, Y)

print(X_train[0:6,:])
print(y_train[0:6])
input_dim = X_train.shape[1]

rf_clf = RandomForestClassifier(n_estimators = 500)  # Modifique aqui os hyperparâmetros
rf_clf.fit(X_train, y_train)
rf_pred_class = rf_clf.predict(X_val)
rf_pred_scores = rf_clf.predict_proba(X_val)
accuracy, recall, precision, f1, auroc, aupr = Utils.compute_performance_metrics(y_val, rf_pred_class, rf_pred_scores)
Utils.print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)