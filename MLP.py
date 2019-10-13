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
from keras.optimizers import SGD, Nadam

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from keras.wrappers.scikit_learn import KerasClassifier
"""
training_data = Utils.load_database('data/BASE-PREPROCESSED(TRAIN).gz')

training_data = training_data[:, :-1]

pagantes, inadiplentes  = Utils.separate_classes(training_data)

training_data = Utils.replicate_shuffle_merge(pagantes, inadiplentes)

X, Y = Utils.get_input_output(training_data)

X_train, y_train, X_test, y_test, X_val, y_val = Utils.separate_train_test_eval(X, Y)

print(X_train)

X_train, X_test, X_val = Utils.normalize(X_train, X_test, X_val)
"""
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

db_train = Utils.load_database('data/TRAIN_PREPROC')
db_test = Utils.load_database('data/TEST_PREPROC')
db_val = Utils.load_database('data/VAL_PREPROC')

X_train, y_train, X_test, y_test, X_val, y_val = (db_train[:, :-1], db_train[:, -1],
                                                  db_test[:, :-1], db_test[:, -1],
                                                  db_val[:, :-1], db_val[:, -1])

del db_train, db_test, db_val
#y_train = np.where((y_train == 0), y_train, -1)
#y_test = np.where((y_test == 0), y_test, -1)
#y_val = np.where((y_val == 0), y_val, -1)
# Número de features do nosso data set.
print(X_train[0:6,:])
print(y_train[0:6   ])
input_dim = X_train.shape[1]
print(input_dim)

def create_squared_vars(array):
   l = np.zeros(array.shape)
   i = 0
   for x in array:
      j = 0
      for v in x:
         sq = v*v
         l[i][j] = sq
         j += 1
      i += 1
   ret = np.concatenate((array, l), axis = 1)
   return ret

X_train = create_squared_vars(X_train)
X_test = create_squared_vars(X_test)
X_val = create_squared_vars(X_val)
input_dim = X_train.shape[1]
print(X_train.shape)


def create_baseline_model(lr):
   # Aqui criamos o esboço da rede.
   classifier = Sequential()
   # Agora adicionamos a camada de entrada contendo 312 neurônios e função de ativação
   # tangente hiperbólica. Por ser a primeira camada adicionada à rede, precisamos especificar
   # a dimensão de entrada (número de features do data set).
   classifier.add(Dense(X_train.shape[1], input_dim=input_dim))
   classifier.add(Dense(39, kernel_initializer='normal', activation='sigmoid'))
   classifier.add(Dense(78, kernel_initializer='normal', activation='relu'))
   # Agora adicionamos a primeira camada escondida contendo 624 neurônios e 
   # função de ativação sigmoid.
   classifier.add(Dense(624, kernel_initializer='normal', activation='sigmoid'))
   classifier.add(Dense(624, kernel_initializer='normal', activation='sigmoid'))
   # Agora adicionamos a segunda camada escondida contendo 156 neurônios e 
   # função de ativação sigmoid.
   classifier.add(Dense(156, kernel_initializer='normal', activation='sigmoid'))
   classifier.add(Dense(78, kernel_initializer='normal', activation='relu'))
   classifier.add(Dense(39, kernel_initializer='normal', activation='sigmoid'))
   # Em seguida adicionamos a camada de saída. Como nosso problema é binário só precisamos de
   # 1 neurônio com função de ativação sigmoidal. A partir da segunda camada adicionada keras já
   # consegue inferir o número de neurônios de entrada (16) e nós não precisamos mais especificar.
   classifier.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
   
   opt = Nadam(lr=lr)
   classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
   return classifier

# Por fim compilamos o modelo especificando um otimizador, a função de custo, e opcionalmente
# métricas para serem observadas durante treinamento.
for lambd in [0.002, 0.01, 0.05, 0.1, 1.0, 3.0]:
   print("Executing For Learning Rate: " + str(lambd))
   classifier = create_baseline_model(lambd)
   
   # Para treinar a rede passamos o conjunto de treinamento e especificamos o tamanho do mini-batch,
   # o número máximo de épocas, e opcionalmente callbacks. No seguinte exemplo utilizamos early
   # stopping para interromper o treinamento caso a performance não melhore em um conjunto de validação.
   history = classifier.fit(X_train, y_train, batch_size=64, epochs=200, 
                            callbacks=[EarlyStopping(patience=3)], validation_data=(X_val, y_val))
   
   Utils.plot_training_error_curves(history)
   
   # Fazer predições no conjunto de teste
   y_pred_scores = classifier.predict(X_test)
   y_pred_class = classifier.predict_classes(X_test, verbose=0)
   #print(y_pred_scores[:50, :])
   #print(y_pred_class[:50, :])
   y_pred_scores_0 = 1 - y_pred_scores
   y_pred_scores = np.concatenate([y_pred_scores_0, y_pred_scores], axis=1)
   
   ## Matriz de confusão
   print('Matriz de confusão no conjunto de teste:')
   print(confusion_matrix(y_test, y_pred_class))
   
   ## Resumo dos resultados
   losses = Utils.extract_final_losses(history)
   print()
   print("{metric:<18}{value:.4f}".format(metric="Train Loss:", value=losses['train_loss']))
   print("{metric:<18}{value:.4f}".format(metric="Validation Loss:", value=losses['val_loss']))
   print('\nPerformance no conjunto de teste:')
   accuracy, recall, precision, f1, auroc, aupr = Utils.compute_performance_metrics(y_test, y_pred_class, y_pred_scores)
   Utils.print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
