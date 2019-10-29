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
from keras.optimizers import SGD, Adamax, Nadam, Adadelta
from keras.initializers import RandomNormal

from sklearn.metrics import confusion_matrix


training_data = Utils.load_database('data/BASE-PREPROCESSED(TRAIN).gz', sep = '\t')

"""
training_data = np.delete(training_data, range(1, training_data.shape[1] - 1), 1)
print(training_data.shape)
x = training_data[:, -1]
y = training_data[:, 0].reshape(training_data.shape[0], 1)
del training_data
x = np.array([k.strip('][').split(',') for k in x]).astype(float)[:, :-3]
training_data = np.concatenate((y, x), axis= 1)
del x, y
"""
#print(training_data)

#training_data = training_data[:10000, :-1]

#df = pd.DataFrame(training_data)
#file = open('data/SMALL-PREPROCESSED(TRAIN)', 'w+')
#file.write(df.to_csv())
#file.close()

training_data = training_data[:, :-5]
pagantes, inadiplentes  = Utils.separate_classes(training_data)

training_data = Utils.replicate_shuffle_merge(pagantes, inadiplentes)

X_train, y_train = Utils.get_input_output(training_data)

#X_train, y_train, X_test, y_test, X_val, y_val = Utils.separate_train_test_eval(X, Y)

print(X_train.shape)

del training_data

training_data = Utils.load_database('data/BASE-PREPROCESSED(TESTE).gz', sep = '\t')

"""
training_data = np.delete(training_data, range(1, training_data.shape[1] - 1), 1)
print(training_data.shape)
x = training_data[:, -1]
y = training_data[:, 0].reshape(training_data.shape[0], 1)
del training_data
x = np.array([k.strip('][').split(',') for k in x]).astype(float)
training_data = np.concatenate((y, x), axis= 1)
del x, y
"""

training_data = training_data[:, :-2]
X_test, y_test = Utils.get_input_output(training_data)

del training_data

training_data = Utils.load_database('data/BASE-PREPROCESSED(VALIDACAO).gz', sep = '\t')

"""
training_data = np.delete(training_data, range(1, training_data.shape[1] - 1), 1)
print(training_data.shape)
x = training_data[:, -1]
y = training_data[:, 0].reshape(training_data.shape[0], 1)
del training_data
x = np.array([k.strip('][').split(',') for k in x]).astype(float)
training_data = np.concatenate((y, x), axis= 1)
del x, y
"""

training_data = training_data[:, :-2]
X_val, y_val = Utils.get_input_output(training_data)

X_train, X_test, X_val = Utils.normalize(X_train, X_test, X_val)

del training_data

#print(X_train)
#print(y_train.shape)
#print(X_val.shape)
#print(y_val.shape)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
"""
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

#X_train = create_squared_vars(X_train)
#X_test = create_squared_vars(X_test)
#X_val = create_squared_vars(X_val)
"""
#%%
input_dim = X_train.shape[1]
print(X_train.shape)


def create_baseline_model(lr, opt = Nadam(lr = 0.3)):
   # Aqui criamos o esboço da rede.
   classifier = Sequential()
   # Agora adicionamos a camada de entrada contendo 312 neurônios e função de ativação
   # tangente hiperbólica. Por ser a primeira camada adicionada à rede, precisamos especificar
   # a dimensão de entrada (número de features do data set).
   #initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)
   classifier.add(Dense(X_train.shape[1], activation='tanh', input_dim=input_dim))
   #classifier.add(Dense(39, kernel_initializer='normal', activation='sigmoid'))
   #classifier.add(Dense(78, kernel_initializer='normal', activation='relu'))
   # Agora adicionamos a primeira camada escondida contendo 624 neurônios e 
   # função de ativação sigmoid.
   classifier.add(Dense(3120, kernel_initializer='random_uniform', activation='sigmoid'))
   #classifier.add(Dense(624, kernel_initializer='normal', activation='sigmoid'))
   # Agora adicionamos a segunda camada escondida contendo 156 neurônios e 
   # função de ativação sigmoid.
   #classifier.add(Dense(156, kernel_initializer='normal', activation='sigmoid'))
   #classifier.add(Dense(78, kernel_initializer='normal', activation='relu'))
   #classifier.add(Dense(39, kernel_initializer='normal', activation='sigmoid'))
   # Em seguida adicionamos a camada de saída. Como nosso problema é binário só precisamos de
   # 1 neurônio com função de ativação sigmoidal. A partir da segunda camada adicionada keras já
   # consegue inferir o número de neurônios de entrada (16) e nós não precisamos mais especificar.
   classifier.add(Dense(1, kernel_initializer='random_uniform', activation='sigmoid'))
   
   #opt = Nadam(lr=lr)
   classifier.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
   return classifier

# Por fim compilamos o modelo especificando um otimizador, a função de custo, e opcionalmente
# métricas para serem observadas durante treinamento.
#lambd = 0.002
add = 0
for optm in ["Nadam", "Adamax", "Adadelta"]:        
    for lambd in [0.001, 0.0015, 0.0013, 0.0008]:#, 0.05, 0.1, 0.5, 1.0, 3.0]:
        optmizer = None
        if (optm == "Nadam"):
            optmizer = Nadam(lr = lambd)
        elif (optm == "Adamax"):
            optmizer = Adamax(lr = lambd)
        elif (optm == "Adadelta" and add == 0):
            optmizer = Adadelta()
            lambd = 1.0
            add = 1
        else:
            continue
        print("Executing " + optm +  " Optimizer For Learning Rate: " + str(lambd))
        classifier = create_baseline_model(lambd, optmizer)
       
        # Para treinar a rede passamos o conjunto de treinamento e especificamos o tamanho do mini-batch,
        # o número máximo de épocas, e opcionalmente callbacks. No seguinte exemplo utilizamos early
        # stopping para interromper o treinamento caso a performance não melhore em um conjunto de validação.
        #callbacks=[EarlyStopping(patience=3)]
        history = classifier.fit(X_train, y_train, batch_size=64, epochs=150, verbose = 2,
                                callbacks=[EarlyStopping(patience=25)], 
                                validation_data=(X_val, y_val))
       
        Utils.plot_training_error_curves(history)
       
        # Fazer predições no conjunto de teste
        y_pred_scores = classifier.predict(X_test)
        y_pred_class = classifier.predict_classes(X_test, verbose=1).ravel()
        #print(y_pred_scores[:50, :])
        #print(y_pred_class[:50, :])
        y_test = list(y_test)
        y_pred_scores_0 = 1 - y_pred_scores
        y_pred_scores = np.concatenate([y_pred_scores_0, y_pred_scores], axis=1)
        
        ## Matriz de confusão
        print('Matriz de confusão no conjunto de teste:')
        #print(confusion_matrix(list(y_test), y_pred_class))
        Utils.print_binary_confusion(list(y_test), y_pred_class)
        
        ## Resumo dos resultados
        losses = Utils.extract_final_losses(history)
        print()
        print("{metric:<18}{value:.4f}".format(metric="Train Loss:", value=losses['train_loss']))
        print("{metric:<18}{value:.4f}".format(metric="Validation Loss:", value=losses['val_loss']))
        print('\nPerformance no conjunto de teste:')
        accuracy, recall, precision, f1, auroc, aupr = Utils.compute_performance_metrics(y_test, y_pred_class, y_pred_scores)
        Utils.print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
    
#%%