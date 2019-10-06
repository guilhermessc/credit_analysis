print("importing")
import numpy as np
import pandas as pd
import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier

import scikitplot as skplt
import matplotlib
import matplotlib.pyplot as plt

def shuffle_array(array):
   np.random.shuffle(array)
   
def load_database(filename = 'data/BASE-PREPROCESSED(TRAIN).gz'):
   start_time=datetime.datetime.now()
   
   data = pd.read_table(filename)
   data.drop_duplicates(inplace=True)
   
   end_time=datetime.datetime.now()
   print("Loading time taken - {}".format(end_time-start_time))
   return data.values
   
def separate_classes(database):
   start_time=datetime.datetime.now()
   
   training_preproc_neg = np.array([x for x in database if x[0] == 0])
   training_preproc_pos = np.array([x for x in database if x[0] == 1])
   
   end_time=datetime.datetime.now()
   print("Separation time taken - {}".format(end_time-start_time))
   return (training_preproc_pos, training_preproc_neg)

def replicate_shuffle_merge(major_class_samples, minor_class_samples):
   start_time=datetime.datetime.now()
   
   prop = major_class_samples.shape[0] // minor_class_samples.shape[0]
   training_neg = np.tile(minor_class_samples, (prop,1))
   
   end_time=datetime.datetime.now()
   print("Replication time taken - {}".format(end_time-start_time))
   start_time=datetime.datetime.now()
   
   shuffle_array(training_neg)
   dataset = np.concatenate((training_neg, major_class_samples), axis = 0)
   shuffle_array(dataset)
   
   end_time=datetime.datetime.now()
   print("Shuffle time taken - {}".format(end_time-start_time))
   return dataset

def get_input_output(database):
   x = database[:, 1:]
   y = database[:, 0]
   return (x, y)
   
def separate_train_test_eval(X, Y):
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, 
                                                    random_state=42, stratify=y)
   X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/3, 
                                                  random_state=42, stratify=y_train)
   return (X_train, y_train, X_test, y_test, X_val, y_val)
   
def normalize(database):
   scaler = StandardScaler()
   db = scaler.fit_transform(database)
   return db


## Provided utilities functions
def extract_final_losses(history):
    """Função para extrair o melhor loss de treino e validação.
    
    Argumento(s):
    history -- Objeto retornado pela função fit do keras.
    
    Retorno:
    Dicionário contendo o melhor loss de treino e de validação baseado 
    no menor loss de validação.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    idx_min_val_loss = np.argmin(val_loss)
    return {'train_loss': train_loss[idx_min_val_loss], 'val_loss': val_loss[idx_min_val_loss]}

def plot_training_error_curves(history):
    """Função para plotar as curvas de erro do treinamento da rede neural.
    
    Argumento(s):
    history -- Objeto retornado pela função fit do keras.
    
    Retorno:
    A função gera o gráfico do treino da rede e retorna None.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    fig, ax = plt.subplots()
    ax.plot(train_loss, label='Train')
    ax.plot(val_loss, label='Validation')
    ax.set(title='Training and Validation Error Curves', xlabel='Epochs', ylabel='Loss (MSE)')
    ax.legend()
    plt.show()

def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        skplt.metrics.plot_ks_statistic(y, y_pred_scores)
        plt.show()
        y_pred_scores = y_pred_scores[:, 1]
        auroc = roc_auc_score(y, y_pred_scores)
        aupr = average_precision_score(y, y_pred_scores)
        performance_metrics = performance_metrics + (auroc, aupr)
    return performance_metrics

def print_metrics_summary(accuracy, recall, precision, f1, auroc=None, aupr=None):
    print()
    print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))
    print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))
    print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision))
    print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))
    if auroc is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))
    if aupr is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))
