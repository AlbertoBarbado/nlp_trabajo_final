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
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
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


def rf_polarity_train(dominio):
    
    df_final = obtain_train_corpus()
    # Puedo separarlo en distintos df segun el dominio
    df_domain_total = [{category:df_domain} for category, df_domain in df_final.groupby('category')]
    
    # Cargo los encodings
    with open(os.path.abspath('') + r'/generated_data/encodings.p', 'rb') as handle:
            encodings = pickle.load(handle)
    polarity = encodings['polarity']
    
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
                
                y = df['polarity']
                y = np.array([polarity[i] for i in y])
                                
#                # Encoding a numerico
#                labelencoder_X = LabelEncoder()
#                y=labelencoder_X.fit_transform(y) # Codifico en valores numericos las clases que hay
#                y_original = y
#                
#                # Si tengo un vector donde el mayor valor no sea 1 (y que no sea todo 0's)
#                if max(y_original) != 1 and sum(y_original) != 0:
#                    # Encoding a one-hot
#                    y = y.reshape(-1, 1)
#                    onehotencoder = OneHotEncoder()
#                    y = onehotencoder.fit_transform(y).toarray()
#                    
#                    # Remove dummy variable trap
#                    y = y[:, 1:] # Elimino una de las columnas por ser linearmente dependiente de las demas
                
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

                
                # Deshago el onehot encoding para sacar las metricas
                y_val = pd.DataFrame(y_val)
                y_pred = pd.DataFrame(y_pred)       
                y_val = [(np.argmax(np.asarray(x)) + 1.0) if max(np.asarray(x)) > 0 else 0.0 for x in y_val.values.tolist()]
                y_pred = [(np.argmax(np.asarray(x)) + 1.0) if max(np.asarray(x)) > 0 else 0.0 for x in y_pred.values.tolist()]

                        
                # Making the Confusion Matrix
                cm = confusion_matrix(y_val, y_pred)
                
                # Accuracy
                accuracy = accuracy_score(y_val, y_pred)
                
                # Precision
                average_precision = precision_score(y_val, y_pred, average = "weighted")
                
                # Recall
                recall = recall_score(y_val, y_pred, average='weighted')
                
                # F1
                f1 = f1_score(y_val, y_pred, average='weighted', labels=np.unique(y_pred))
                
                print("Modelo "+str(i)+" resultados")
                print("accuracy ", accuracy, " precision ", average_precision, " recall ", recall, " f1 ", f1)
 
                
                # Eliminio los backslashes de las palabras que los tengan
                if '/' in entidad:
                        entidad= entidad.replace('/', '_')
                         
                # Persistencia del modelo entrenado
                with open(os.path.abspath('') + r'/generated_data/model_rf_polarity_'+str(entidad)+'.p', 'wb') as handle:
                        pickle.dump(classifier, handle)

                print("Modelo "+ str(i)+" entrenado y guardado")
                i += 1
                


def cnn_rnn_polarity_train(dominio):

    df_final = obtain_train_corpus()
    # Puedo separarlo en distintos df segun el dominio
    df_domain_total = [{category:df_domain} for category, df_domain in df_final.groupby('category')]
    
    # Cargo los encodings
    with open(os.path.abspath('') + r'/generated_data/encodings.p', 'rb') as handle:
            encodings = pickle.load(handle)
    polarity = encodings['polarity']
    topic_priority = encodings['topic_priority']
    
    
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
                
                
                # Saco la mediana del tamaño de las secuencias
                with open(os.path.abspath('') + r'/generated_data/max_length.p', 'rb') as handle:
                    dict_max = pickle.load(handle)   

                
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
                         pickle.dump(data, handle)
                        
                    print("word2idx creado y guardado para ", entidad)
                    
                    
                # Feature scaling de los datos de entrada
                X = [preprocessing.scale(x) for x in X]
                   
                # Labeling numérico de las clases
                y = list(df['polarity'])
#                y = np.array([polarity[i] for i in y])
                
                
                labelencoder_Y = LabelEncoder()
                Y = labelencoder_Y.fit_transform(y)   
                                
#                Y = np.array([np.array([i[0], i[1], 0]) if i[0] != i[1] else np.array([i[0], i[1], 1]) for i in Y])
                
                
                M = 50 # tamaño hidden layer
                # V # tamaño del vocabulario
                K = len(set(Y)) # Numero de clases
                
#                seq_length = 30
                
                
                seq_length = int(np.floor(dict_max[entidad]))
                
                # Defino el dataset de validacion, especifico su tamaño y reservo esa cantidad de datos para ello 
                X, Y = shuffle(X, Y)
                X_b = X
                Y_b = Y
                
                N = len(X)
                Nvalid = round(N/5)
                Xvalid, Yvalid = X[-Nvalid:], Y[-Nvalid:] # Datos que dejo para validar
                X, Y = X[:-Nvalid], Y[:-Nvalid] # Datos que dejo para entrenar
                
                # Me aseguro que tengo todas las clases en el train/valid 
                # - sino cojo todos los datos por defecto y no hago validation
                """En este caso significa que hay pocos datos, no puedo hacer un split a validacion; los datos
                de validacion seran irrelevantes"""
                if len(set(Y)) != K or len(set(Yvalid)) != K:
                    X = X_b
                    Y = Y_b
                    Xvalid = X_b
                    Yvalid = Y

    
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
                model.add(Dense(K, activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                print(model.summary())
                model.fit(X_train, y_train, epochs=3, batch_size=64)
                
                # Evaluación final del modelo
                scores = model.evaluate(X_valid, y_valid, verbose=0)
                print("Accuracy: %.2f%%" % (scores[1]*100))                
                
                
                # Prediciones
                y_pred = model.predict_classes(X_valid)
                
                # Deshago el onehot encoding para sacar las metricas
                y_val = y_valid
                
                y_val = pd.DataFrame(y_val)
#                y_pred = pd.DataFrame(y_pred)       
                y_val = [(np.argmax(np.asarray(x))) if max(np.asarray(x)) > 0 else 0.0 for x in y_val.values.tolist()]
#                y_pred = [(np.argmax(np.asarray(x))) if max(np.asarray(x)) > 0 else 0.0 for x in y_pred.values.tolist()]
                
                y_pred = list(y_pred)
                
                # CM
                confusion = confusion_matrix(np.argmax(y_valid, axis=1), y_pred)
                
                # Accuracy
                accuracy = accuracy_score(y_val, y_pred)
                
                # Precision
                precision = precision_score(y_val, y_pred, average = "weighted")
                
                # Recall
                recall = recall_score(y_val, y_pred, average='weighted')
                
                # F1
                f1 = f1_score(y_val, y_pred, average='weighted', labels=np.unique(y_pred))
                
                print("Modelo "+str(i)+" resultados")
                print("accuracy ", accuracy, " precision ", precision, " recall ", recall, " f1 ", f1)
                
                # Guardo el modelo
                model.save(os.path.abspath('') + "/generated_data/cnn_keras_model_polarity_"+str(entidad)+".h5")
                del(model)
                i += 1
                
                
                
                 
def model_test_eval_rnn(model, df, corpus, entidad):
    
    X = list(df['text'])
                
    print("Entidad: ", entidad)
    
    # Saco la mediana del tamaño de las secuencias
    with open(os.path.abspath('') + r'/generated_data/max_length.p', 'rb') as handle:
        dict_max = pickle.load(handle)   

    
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
        print ("No se han podido cargar los datos")
        return ""
    
    # Labeling numérico de las clases
    y = list(df['polarity'])
        
    # Me aseguro de que no hay utterances vacias
    for i in X:
        if len(i) == 0:
            #print(X.index(i))
            del y[X.index(i)]
            X.remove(i)
        
    # Feature scaling de los datos de entrada
    X = [preprocessing.scale(x) for x in X]


    labelencoder_Y = LabelEncoder()
    Y = labelencoder_Y.fit_transform(y)   
    
    K = len(set(Y)) # Numero de clases
    
#    seq_length = 30
    seq_length = int(np.floor(dict_max[entidad]))
    
    # Defino el dataset de validacion, especifico su tamaño y reservo esa cantidad de datos para ello 
    X, Y = shuffle(X, Y)

    Xvalid, Yvalid = X, Y # Datos que dejo para validar
    
    # Hago el padding/truncado de los datos
    max_review_length = seq_length
    X_valid = sequence.pad_sequences(Xvalid, maxlen=max_review_length) 
    y_valid = Yvalid
    
    top_words = len(word2idx) # palabras del vocabulario
    
    # Pongo los datos de y de forma categórica
    y_valid = to_categorical(y_valid)


    # Evaluación final del modelo
    #scores = model.evaluate(X_valid, y_valid, verbose=0)
    #print("Accuracy: %.2f%%" % (scores[1]*100))                
    
    
    # Prediciones
    y_pred = model.predict_classes(X_valid)
    
    # Deshago el onehot encoding para sacar las metricas
#    y_val = y_valid
#    
#    y_val = pd.DataFrame(y_val)
#    y_pred = pd.DataFrame(y_pred)     
    
    y_val = Yvalid
    
#    y_val = [(np.argmax(np.asarray(x))) if max(np.asarray(x)) > 0 else 0.0 for x in y_val]
#    y_pred = [(np.argmax(np.asarray(x))) if max(np.asarray(x)) > 0 else 0.0 for x in y_pred.values.tolist()]

    
    # CM
    confusion = confusion_matrix(Yvalid, y_pred)
    
    # Accuracy
    accuracy = accuracy_score(Yvalid, y_pred)
    
    # Precision
    precision = precision_score(Yvalid, y_pred, average = "weighted")
    
    # Recall
    recall = recall_score(Yvalid, y_pred, average='weighted')
    
    # F1
    f1 = f1_score(y_val, y_pred, average='weighted', labels=np.unique(y_pred))
    
    print("Resultados")
    print("accuracy ", accuracy, " precision ", precision, " recall ", recall, " f1 ", f1)
    
    return confusion, accuracy, precision, recall, f1


def chunks_polarity():
    
    ##### Using chunks
    # nltk.download('words')
    my_sent = "WASHINGTON -- In the wake of a string of abuses by New York police officers in the 1990s, Loretta E. Lynch, the top federal prosecutor in Brooklyn, spoke forcefully about the pain of a broken trust that African-Americans felt and said the responsibility for repairing generations of miscommunication and mistrust fell to law enforcement."
    parse_tree = nltk.ne_chunk(nltk.tag.pos_tag(my_sent.split()), binary=True)  # POS tagging before chunking!
    
    
    named_entities = []
    
    for t in parse_tree.subtrees():
        if t.label() == 'NE':
            named_entities.append(t)
            # named_entities.append(list(t))  # if you want to save a list of tagged words instead of a tree
    
    print (named_entities)
    
    
    df_final = obtain_train_corpus()
    # Puedo separarlo en distintos df segun el dominio
    df_domain_total = [{category:df_domain} for category, df_domain in df_final.groupby('category')]
    
    # Cargo los encodings
    with open(os.path.abspath('') + r'/generated_data/encodings.p', 'rb') as handle:
            encodings = pickle.load(handle)
    polarity = encodings['polarity']
    
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
            
            y = df['polarity']
            y = np.array([polarity[i] for i in y])
            
            
            # POS + Chunking
            
            for sent in X:
                parse_tree = nltk.ne_chunk(nltk.tag.pos_tag(sent.split()), binary=True)  # POS tagging before chunking!
     
        
    
           
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
    
    
    
    
from scipy.optimize import minimize
    
def polarity_hp_tuning():
    
    # Funcion objetivo a minimizar
    def objective(x):
        n_trees = x[0]
        n_trees = int(np.floor(n_trees))
        res = rf_polarity_train_tuning(n_trees)
        print ("trees: ", n_trees,"f1: ", 1-res)
        return res
    
    # Constraints
    # None
    
    # Bounds 

    b = (2,300) # rango de bounds
    bnds = (b, b) # bounds para el numero de arboles
    
    # Initial guesses
    x0 = [150, 150]

    # Solution
    sol = minimize(objective, x0, method='L-BFGS-B', bounds=bnds, options={'maxiter':10, 'eps':10})
    
    print(sol.x)
    
    n_trees = sol.x[0]
    print(n_trees)
    
    return n_trees