# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 17:25:27 2019

@author: brayner

Model: MLP
"""
import Utils
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
from keras.wrappers.scikit_learn import KerasClassifier

training_data = Utils.load_database('data/BASE-PREPROCESSED(TRAIN).gz')

pagantes, inadiplentes  = Utils.separate_classes(training_data)

training_data = Utils.replicate_shuffle_merge(pagantes, inadiplentes)

X, Y = Utils.get_input_output(training_data)

X_train, y_train, X_test, y_test, X_val, y_val = Utils.separate_train_test_eval()


X_train, X_test, X_val = Utils.normalize(X_train), Utils.normalize(X_test), Utils.normalize(X_val)

# Número de features do nosso data set.
input_dim = X_train.shape[1]

# Aqui criamos o esboço da rede.
classifier = Sequential()

# Agora adicionamos a camada de entrada contendo 312 neurônios e função de ativação
# tangente hiperbólica. Por ser a primeira camada adicionada à rede, precisamos especificar
# a dimensão de entrada (número de features do data set).
classifier.add(Dense(312, activation='tanh', input_dim=input_dim))

# Agora adicionamos a primeira camada escondida contendo 624 neurônios e 
# função de ativação sigmoid.
classifier.add(Dense(624, activation='sigmoid'))

# Agora adicionamos a segunda camada escondida contendo 156 neurônios e 
# função de ativação sigmoid.
classifier.add(Dense(156, activation='sigmoid'))

# Em seguida adicionamos a camada de saída. Como nosso problema é binário só precisamos de
# 1 neurônio com função de ativação sigmoidal. A partir da segunda camada adicionada keras já
# consegue inferir o número de neurônios de entrada (16) e nós não precisamos mais especificar.
classifier.add(Dense(1, activation='sigmoid'))

# Por fim compilamos o modelo especificando um otimizador, a função de custo, e opcionalmente
# métricas para serem observadas durante treinamento.
classifier.compile(optimizer='adam', loss='mean_squared_error')


# Para treinar a rede passamos o conjunto de treinamento e especificamos o tamanho do mini-batch,
# o número máximo de épocas, e opcionalmente callbacks. No seguinte exemplo utilizamos early
# stopping para interromper o treinamento caso a performance não melhore em um conjunto de validação.
history = classifier.fit(X_train, y_train, batch_size=64, epochs=100000, 
                         callbacks=[EarlyStopping(patience=3)], validation_data=(X_val, y_val))

Utils.plot_training_error_curves(history)