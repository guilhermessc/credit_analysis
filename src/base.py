import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score

import scikitplot as skplt
import matplotlib
import matplotlib.pyplot as plt
print(pd)
training_preproc_dataset_pand = pd.read_table('../data/BASE-PREPROCESSED(TRAIN).gz')
training_preproc_dataset_pand.drop_duplicates(inplace=True)  # Remove exemplos repetidos

training_preproc_dataset = training_preproc_dataset_pand.values
training_preproc_neg = np.array([x for x in training_preproc_dataset if x[0] == 0])
training_preproc_pos = np.array([x for x in training_preproc_dataset if x[0] == 1])
prop = len(training_preproc_pos) // len(training_preproc_neg)
print(prop)
print("Before Replication:")
print("inad: " + str(len(training_preproc_neg)) + " ; pag: " + str(len(training_preproc_pos)))
training_preproc_neg = prop * training_preproc_neg
print("After Replication:")
print("inad: " + str(len(training_preproc_neg)) + " ; pag: " + str(len(training_preproc_pos)))

print(training_preproc_neg.shape)
print(training_preproc_neg[0:5, 0:5])
np.random.shuffle(training_preproc_neg)
training_preproc_dataset = training_preproc_neg + training_preproc_pos
print(training_preproc_dataset[0:5, 0:5])
np.random.shuffle(training_preproc_dataset)
#X = data_set.iloc[:, :-1].values
#y = data_set.iloc[:, -1].values
#y = np.where(y == -1, 0, 1)

      

print(Tot[0:5, 0:5])
# Exibe as 5 primeiras linhas do data set
print(X.shape)
print(X[0:5])
print(y[0:5])