# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 19:23:41 2018

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
from other_preprocessing import word_tf_idf

global stemmer
stemmer = SnowballStemmer("english")


def rf_priority_train(df, dominio):
    
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
                y = list(df['topic_priority'])
                
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
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 0)
                
                
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
                                
                    y_val  = [np.argmax(np.asarray(x)) for x in y_val.values.tolist()]
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
                print("accuracy ", accuracy, " precision ", average_precision, " recall ", recall) # Se ve que los resultados son muy malos
 
                
                # Eliminio los backslashes de las palabras que los tengan
                if '/' in entidad:
                        entidad= entidad.replace('/', '_')
                         
                # Persistencia del modelo entrenado
                with open(os.path.abspath('') + r'/generated_data/model_rf_priority_detection_'+str(entidad)+'.p', 'wb') as handle:
                        pickle.dump(vocabulario, handle)

                print("Modelo "+ str(i)+" entrenado y guardado")
                i += 1