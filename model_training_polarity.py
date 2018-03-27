# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:19:42 2018

@author: alber
"""

import os
import glob
import pandas as pd
import re
import numpy as np
import pickle

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model
from keras.utils.np_utils import to_categorical

from data_preprocessing import word_vector, corpus_generation, obtain_train_corpus, word2idx_creation, word2idx_conversion

global stemmer
stemmer = SnowballStemmer("english")


def rf_polarity_train(df, dominio):
    
    df_final = obtain_train_corpus()
    # Puedo separarlo en distintos df segun el dominio
    df_domain_total = [{category:df_domain} for category, df_domain in df_final.groupby('category')]
    
    if dominio == "entidad":
        # Tambien puedo separar a nivel de dominio y entity
        df_domain_total_entity = {}
        for df in df_domain_total:
            category = list(df.keys())[0]
            df = list(df.values())[0]
            df_entities = [{entity:df_entity} for entity, df_entity in df.groupby('entity_name')]
            df_domain_total_entity.update({category:df_entities})
            
        vocabulario = corpus_generation(df_domain_total_entity, "entidad")
        entidades = list(vocabulario.keys())
        categorias = list(df_domain_total_entity.keys())
         
        i = 1
        total = len(entidades)
        for categoria in categorias:
            for df in df_domain_total_entity[categoria]:
                
                print("Entrendando modelo " +  str(i) + "/" + str(total))
                
                entidad = list(df.keys())[0]
                df = list(df.values())[0]
                corpus = vocabulario[entidad][0]
                
                print("Entidad: ", entidad)
                 
                X = list(df['text'])
                y = list(df['polarity'])
                
                # Encoding a numerico
                labelencoder_X = LabelEncoder()
                y=labelencoder_X.fit_transform(y) # Codifico en valores numericos las clases que hay
                y_original = y
                
                if max(y_original) != 1:
                    # Encoding a one-hot
                    y = y.reshape(-1, 1)
                    onehotencoder = OneHotEncoder()
                    y = onehotencoder.fit_transform(y).toarray()
                    
                    # Remove dummy variable trap
                    y = y[:, 1:] # Elimino una de las columnas por ser linearmente dependiente de las demas
                
                # Encoding numerico de las palabras de los vectores de entrada segun el vocabulario
                
                Xt = word_vector(X, corpus)
                X = Xt
                
                # Train/validation split
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)
                
                
                # Por ser un RF no hace falta hacer feature scaling
                """from sklearn.preprocessing import StandardScaler
                sc_X = StandardScaler()
                X_train = sc_X.fit_transform(X_train)
                X_test = sc_X.transform(X_test)
                sc_y = StandardScaler()
                y_train = sc_y.fit_transform(y_train)"""
                
                # Fitting Random Forest Classificator to the Training set
                classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
                classifier.fit(X_train, y_train)
                print("Entrenamiento terminado")
                
                # Predicting the Test set results
                y_pred = classifier.predict(X_val)
                
                if max(y_original) != 1:
                    # Formatting results
                    y_val_original = np.asarray(y_val)
                    y_val = pd.DataFrame(y_val)
                    y_pred = pd.DataFrame(y_pred)
                                
                    y_val = [np.argmax(np.asarray(x)) for x in y_val.values.tolist()]
                    y_pred = [np.argmax(np.asarray(x)) for x in y_pred.values.tolist()]              
                            
                # Making the Confusion Matrix
                cm = confusion_matrix(y_val, y_pred)
                
                # Accuracy
                accuracy = accuracy_score(y_val, y_pred)
                
                # Precision
                average_precision = precision_score(y_val, y_pred, average = "macro")
                
                # Recall
                recall = recall_score(y_val, y_pred, average='macro')
                
                print("Modelo "+str(i)+" resultados")
                print("accuracy ", accuracy, " precision ", average_precision, " recall ", recall)
 
                
                # Eliminio los backslashes de las palabras que los tengan
                if '/' in entidad:
                        entidad= entidad.replace('/', '_')
                         
                # Persistencia del modelo entrenado
                with open(os.path.abspath('') + r'/generated_data/model_rf_polarity_'+str(entidad)+'.p', 'wb') as handle:
                        pickle.dump(vocabulario, handle)

                print("Modelo "+ str(i)+" entrenado y guardado")
                i += 1
                


def cnn_rnn_polarity_train(df, dominio):

    df_final = obtain_train_corpus()
    # Puedo separarlo en distintos df segun el dominio
    df_domain_total = [{category:df_domain} for category, df_domain in df_final.groupby('category')]
    
    if dominio == "entidad":
        # Tambien puedo separar a nivel de dominio y entity
        df_domain_total_entity = {}
        for df in df_domain_total:
            category = list(df.keys())[0]
            df = list(df.values())[0]
            df_entities = [{entity:df_entity} for entity, df_entity in df.groupby('entity_name')]
            df_domain_total_entity.update({category:df_entities})
            
        vocabulario = corpus_generation(df_domain_total_entity, "entidad")
        entidades = list(vocabulario.keys())
        categorias = list(df_domain_total_entity.keys())
         
        # Creacion del word2idx - si ya existiese lo cargo del disco
        i = 1
        total = len(entidades)
        for categoria in categorias:
            for df in df_domain_total_entity[categoria]:
                
                print("Entrendando modelo " +  str(i) + "/" + str(total))
                
                entidad = list(df.keys())[0]
                df = list(df.values())[0]
                corpus = vocabulario[entidad][0]
                
                X = list(df['text'])
                
                print("Entidad: ", entidad)
                
                # Eliminio los backslashes de las palabras que los tengan
                if '/' in entidad:
                        entidad= entidad.replace('/', '_')
                
                try:
                    with open(os.path.abspath('') + '/generated_data/word2idx_'+str(entidad)+'.p', 'rb') as handle:
                        data = pickle.load(handle)
                    
                    word2idx = data['word2idx']
                    X = word2idx_conversion(X, corpus, word2idx)
                    print("Word2idx cargado para ", entidad)
                    
                except:
                    print("No existe aun word2idx. Creando el word2idx para ", entidad)
                    word2idx, X = word2idx_creation(X, corpus)
                    data = {'word2idx':word2idx}
                    with open(os.path.abspath('') + '/generated_data/word2idx_'+str(entidad)+'.p', 'wb') as handle:
                         pickle.dump(vocabulario, handle)
                        
                    print("word2idx creado y guardado para ", entidad)
                    
                    
                # Feature scaling de los datos de entrada
                X = [preprocessing.scale(x) for x in X]

                
                # Labeling numérico de las clases
                y = list(df['polarity'])
                labelencoder_Y = LabelEncoder()
                Y = labelencoder_Y.fit_transform(y)   
                
                M = 50 # tamaño hidden layer
                # V # tamaño del vocabulario
                K = len(set(Y)) # Numero de clases
                
                seq_length = 30
                
                # Defino el dataset de validacion, especifico su tamaño y reservo esa cantidad de datos para ello 
                X, Y = shuffle(X, Y)
                N = len(X)
                Nvalid = round(N/5)
                Xvalid, Yvalid = X[-Nvalid:], Y[-Nvalid:] # Datos que dejo para validar
                X, Y = X[:-Nvalid], Y[:-Nvalid] # Datos que dejo para entrenar
    
                # Hago el padding/truncado de los datos
                max_review_length = seq_length
                X_train = sequence.pad_sequences(X, maxlen=max_review_length) 
                X_valid = sequence.pad_sequences(Xvalid, maxlen=max_review_length) 
                y_train = Y
                y_valid = Yvalid
                
                top_words = len(word2idx) # palabras del vocabulario
                
                # Pongo los datos de y de forma categórica
                y_train = to_categorical(y_train)
                y_valid = to_categorical(y_valid)
                
                # Creo el modelo
                embedding_vecor_length = 32 
                
                model = Sequential()
                model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length)) # Vector embedding
                model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
                model.add(MaxPooling1D(pool_size=2))
                model.add(LSTM(100))
                model.add(Dense(3, activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                print(model.summary())
                model.fit(X_train, y_train, epochs=5, batch_size=64)
                
                # Evaluación final del modelo
                scores = model.evaluate(X_valid, y_valid, verbose=0)
                print("Accuracy: %.2f%%" % (scores[1]*100))                
                
                
                # CM
                y_pred = model.predict_classes(X_valid)
                confusion = confusion_matrix(np.argmax(y_valid, axis=1), y_pred)
                
                # Accuracy
                accuracy = accuracy_score(Yvalid, y_pred)
                
                # Precision
                precision = precision_score(Yvalid, y_pred, average = "macro")
                
                # Recall
                recall = recall_score(Yvalid, y_pred, average='macro') 
                
                print("Accuracy: ", accuracy, " Precision: ", precision, " Recall: ", recall)
                
                # Guardo el modelo
                model.save(os.path.abspath('') + "/generated_data/cnn_keras_model_polarity_"+str(entidad)+".h5")
                del(model)
                
                
                
                
                
                