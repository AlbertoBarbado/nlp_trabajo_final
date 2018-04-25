# -*- coding: latin-1 -*-
"""
Created on Tue Apr 10 18:25:07 2018

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

global stemmer
stemmer = SnowballStemmer("english")


def obtain_test_corpus():
    ## Lectura de tweet_info
    try: 
        files_tweet_info = glob.glob(os.path.abspath('') + '/replab2013-dataset/test/tweet_info/*')
    except:
        pass
    
    df_tweet_info = None
    
    for f in files_tweet_info:
        if type(df_tweet_info)==type(None):
            df_tweet_info = pd.read_csv(f, sep='\t', encoding="latin-1")
        else:
            df_tweet_info = pd.concat([df_tweet_info, pd.read_csv(f, sep='\t', encoding="latin-1")])
            
           
    # Se filtra a sólo los datos en EN
    df_tweet_info = df_tweet_info[df_tweet_info["language"]=="EN"]
        
    
    ## Lectura de etiquetas
    try: 
        files_labeled = glob.glob(os.path.abspath('') + '/replab2013-dataset/test/labeled/*')
    except:
        pass
    
    df_labeled = None
    
    for f in files_labeled:
        if type(df_labeled)==type(None):
            df_labeled = pd.read_csv(f, sep='\t', encoding="latin-1")
        else:
            df_labeled = pd.concat([df_labeled, pd.read_csv(f, sep='\t', encoding="latin-1")])
    
    # Se descartan los UNRELATED #!!!! En principio no los voy a usar
    df_labeled = df_labeled[df_labeled["filtering"]=="RELATED"] 
    
    ## Lectura de tweet text
    try: 
        files_text = glob.glob(os.path.abspath('') + '/test/test/tweet_text/*')
    except:
        pass
    
    df_text = None
    
    for f in files_text:
        if type(df_text)==type(None):
            df_text = pd.read_csv(f, sep='\t', encoding="latin-1")
        else:
            df_text = pd.concat([df_text, pd.read_csv(f, sep='\t', encoding="latin-1")]) 
    
    
    ## Lectura de la tabla entities
    try: 
        files_entities = glob.glob(os.path.abspath('') + '/replab2013-dataset/entities/replab2013_entities.tsv')
    except:
        pass

    df_entities = pd.read_csv(files_entities[0], sep='\t', encoding="latin-1")
    
    ## Joins de las tablas
    # Join tweet_info-labeled
    df = df_tweet_info.merge(df_labeled, how="left", on="tweet_id") # Merge donde ahora debo eliminar los NaN de filtering
    
    # Join tweet_info-tweet_text
    df = df.merge(df_text, how="left", on="tweet_id")
    
    # Eliminar los NaN de filtering #!!!! En principio no los voy a usar
    df = df.dropna(subset=["filtering"])
    
    # Elimino las entradas que no tienen textos
    df = df.dropna(subset=["text"])
    
    # Join de la tabla entities
    df = df.merge(df_entities, how="left", on="entity_id")
    
    # Elimino los tweets que son duplicados de otros
    df = df[df["is_near_duplicate_of"]==-1]
    
    # Selecciono solo los campos que interesan
    df_final = df[["tweet_id", "timestamp", "filtering", "polarity", "topic", 
                   "topic_priority", "text", "entity_id", "author", "query", "entity_name", "category"]]
    
    # Elimino filas duplicadas por si hubiese (que de hecho hay)
    df_final = df_final.drop_duplicates(subset="tweet_id")


    return df_final



def word_preparing_test(documento, corpus):
    
    words = []
    word_list = []
     
    # Hago la tokenizacion
    for utterance in documento:
        # Tokenizo cada frase
        w = re.findall(r'\w+', utterance.lower(),flags = re.UNICODE) # Paso a minusculas todo
        words = w
        # Eliminación de las stop_words 
        words = [word for word in words if word not in stopwords.words('english')]
        # Elimino guiones y otros simbolos raros
        words = [word for word in words if not word.isdigit()] # Elimino numeros     
        # Stemming y eliminación de duplicados
        words = [stemmer.stem(w) for w in words]
        
        # Añado las palabras a la lista
        word_list.extend(words)
        
    word_list = [w if w in corpus else 'UNK' for w in word_list]
    
    return word_list




def model_test_eval(corpus, classifier, X, y, scaling):
    
    y_val = y
    X_val = X
    
    if scaling == True:
        pass
    
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
    
    print("Resultados")
    print("accuracy ", accuracy, " precision ", average_precision, " recall ", recall, " f1 ", f1)
    

    return cm, accuracy, average_precision, recall, f1




