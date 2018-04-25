# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:09:15 2018

@author: alber
"""

# Estudio exploratorio
import os
import glob
import pandas as pd
import re


from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
from scipy.stats.mstats import normaltest

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

global stemmer
import pickle
stemmer = SnowballStemmer("english")



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
words_tot['UNK'] = 0

# Creo Vocabulario
words =  sorted(list(set(word_list))) # Ordeno alfabéticamente y elimino duplicados
words.append('UNK') # Palabra por defecto para las palabras desconocidas

# Suma de TF-IDF por entrada
df_suma = df_pattern.groupby(['n_documento'])['tf-idf'].sum()
df_suma = df_suma.reset_index()

# Valor minimo, medio, maximo y std de una entrada
min_tf_idf = df_suma['tf-idf'].min()
mean_tf_idf = df_suma['tf-idf'].mean()
max_tf_idf = df_suma['tf-idf'].max()
std_tf_idf = df_suma['tf-idf'].std()


# Calculos de desviacion con el Median Absolute Error (MAD)
median = df_suma['tf-idf'].median()
diff  = (df_suma['tf-idf'] - median)**2
diff = np.sqrt(diff)
med_abs_deviation = np.median(diff)
modified_z_score = 0.6745 * diff / med_abs_deviation

df_suma['tf-idf'].hist()

normaltest(df_suma['tf-idf'])

plt.hist(df_suma['tf-idf'], 50, normed=1, facecolor='green', alpha=0.75)



# Longitud de las secuencias (caracteres) a nivel general
df_final = df_final.reset_index()
l = [len(df_final['text'].loc[i]) for i in range(len(df_final))]

import statistics
mediana = statistics.median(l)

# Longitud de las secuencias (palabras) a nivel general
def word_processing_desc(utterance):
    
    words = []
    
    # Hago la tokenizacion
    # Tokenizo cada frase
    w = re.findall(r'\w+', utterance.lower(),flags = re.UNICODE) # Paso a minusculas todo
    words = w
    # Eliminación de las stop_words 
    words = [word for word in words if word not in stopwords.words('english')]
    # Elimino guiones y otros simbolos raros
    words = [word for word in words if not word.isdigit()] # Elimino numeros     
    # Stemming y eliminación de duplicados
    words = [stemmer.stem(w) for w in words]

    return words


l = [word_processing_desc(df_final['text'].loc[i]) for i in range(len(df_final))]
l1 = [len(i) for i in l]
mediana = statistics.median(l1)


# Lo obtenemos para cada categoria y dominio
df_final = obtain_train_corpus()
# Puedo separarlo en distintos df segun el dominio
df_domain_total = [{category:df_domain} for category, df_domain in df_final.groupby('category')]

# Cargo los encodings
with open(os.path.abspath('') + r'/generated_data/encodings.p', 'rb') as handle:
    encodings = pickle.load(handle)
polarity = encodings['polarity']
topic_priority = encodings['topic_priority']
    # Tambien puedo separar a nivel de dominio y entity
df_domain_total_entity = {}
for df in df_domain_total:
    category = list(df.keys())[0]
    df = list(df.values())[0]
    df_entities = [{entity:df_entity} for entity, df_entity in df.groupby('entity_name')]
    df_domain_total_entity.update({category:df_entities})

# Cargo el vocabulario generado
vocabulario = load_corpus(dominio)
entidades = list(vocabulario.keys())
categorias = list(df_domain_total_entity.keys())

d = {}
for categoria in categorias:
    for df in df_domain_total_entity[categoria]:
        entidad = list(df.keys())[0]
        df = list(df.values())[0]
        
        df = df.reset_index()
        l = [word_processing_desc(df['text'].loc[i]) for i in range(len(df))]
        l1 = [len(i) for i in l]
        maximum = max(l1)
        
        # Eliminio los backslashes de las palabras que los tengan
        if '/' in entidad:
                entidad= entidad.replace('/', '_')
                        
        d[entidad] = maximum
                
# Guardo ese diccionario
with open(os.path.abspath('') + r'/generated_data/max_length.p', 'wb') as handle:
    pickle.dump(d, handle)        
        