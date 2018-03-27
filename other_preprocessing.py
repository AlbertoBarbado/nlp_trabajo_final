# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 19:51:30 2018

@author: alber
"""
import os
import glob
import pandas as pd
import re


from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
global stemmer
import pickle
stemmer = SnowballStemmer("english")

def word_tf_idf(documento):
    
    words = []
    word_list = []
    
    df_pattern = pd.DataFrame()
    i = 0
     
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
        # Inicializo la bolsa de palabras
        pattern_words = words

        df = pd.DataFrame(pattern_words)
        df['ocurrencias'] = 1 
        df.columns = ['palabras', 'ocurrencias']
        df = df.groupby(['palabras'])['ocurrencias'].sum() # En este pundo, al pasarlo a indices, se ordenan
        df = pd.DataFrame(df)
        df['n_documento'] = i
        i += 1
        
        df_pattern = df_pattern.append(df)
        
        # Añado las palabras a la lista
        word_list.extend(words)
        
    # N - num de documentos totales
    n = i
    
    df_pattern = df_pattern.reset_index()
    
    # Doc frequency - documentos que contienen cada palabra
    df_pattern["doc"] = 1
    df = df_pattern.groupby(['palabras', 'n_documento'])['doc'].mean()
    df = df.reset_index()
    df = df_pattern.groupby(['palabras'])['doc'].sum()
    df = df.reset_index()
    df_pattern = df_pattern.drop(['doc'], axis=1)
    
    # TF - frecuencia de palabras por entrada
    df_pattern = df_pattern.merge(df, on="palabras", how="left")
    df_pattern['tf-idf'] = df_pattern.apply(lambda x: np.log(n/x['doc'])*x['ocurrencias'], axis=1)

    # Frecuencia total de terminos en el vocabulario
    df_aux = df_pattern.groupby(['palabras'])['ocurrencias'].sum()
    df_aux = df_aux.reset_index()
    
    
    words_tot = df_pattern.to_dict()
    
    words_tot['palabras'].update({'UNK':0}) 

    # Creo Vocabulario
    words =  sorted(list(set(word_list))) # Ordeno alfabéticamente y elimino duplicados
    words.append('UNK') # Palabra por defecto para las palabras desconocidas
    
    # Suma de TF-IDF por entrada
    df_suma = df_pattern.groupby(['n_documento'])['tf-idf'].sum()
    df_suma = df_suma.reset_index()
    
    # Valor minimo, medio, maximo y std de una entrada
    median = df_suma['tf-idf'].median()
    
    
    return words, words_tot, median, df_pattern, df_suma