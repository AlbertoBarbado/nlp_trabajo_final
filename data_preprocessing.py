# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 20:22:33 2018

@author: alberto barbado gonzalez
"""

import os
import glob
import pandas as pd
import re

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
global stemmer
import pickle
stemmer = SnowballStemmer("english")

def obtain_train_corpus():
    ## Lectura de tweet_info
    try: 
        files_tweet_info = glob.glob(os.path.abspath('') + '/replab2013-dataset/training/tweet_info/*')
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
        files_labeled = glob.glob(os.path.abspath('') + '/replab2013-dataset/training/labeled/*')
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
        files_text = glob.glob(os.path.abspath('') + '/training/training/tweet_text/*')
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
    
    # Encoding de dos de las columnas
    labelencoder_X = LabelEncoder()
    onehotencoder = OneHotEncoder()
    df_encoding = df_final["polarity"].copy().drop_duplicates()
    df_encoding = df_encoding.reset_index()
    
    y = labelencoder_X.fit_transform(df_encoding["polarity"])
    y = y.reshape(-1, 1)
    y = onehotencoder.fit_transform(y).toarray()
    
    # Remove dummy variable trap
    y = y[:, 1:] # Elimino una de las columnas por ser linearmente dependiente de las demas
    
    polarity = {}
    [polarity.update({i:j}) for i,j in zip(df_encoding["polarity"], y)]
        
    
    df_encoding = df_final["topic_priority"].copy().drop_duplicates()
    df_encoding = df_encoding.reset_index()
    y = labelencoder_X.fit_transform(df_encoding["topic_priority"])
    y = y.reshape(-1, 1)
    y = onehotencoder.fit_transform(y).toarray()
    
    # Remove dummy variable trap
    y = y[:, 1:] # Elimino una de las columnas por ser linearmente dependiente de las demas
    
    topic_priority = {}
    [topic_priority.update({i:j}) for i,j in zip(df_encoding["topic_priority"], y)]    
    encodings = {'polarity':polarity, 'topic_priority':topic_priority}
    
    # Guardo en disco los encodings
    with open(os.path.abspath('') + r'/generated_data/encodings.p', 'wb') as handle:
            pickle.dump(encodings, handle)

    return df_final



def word_processing(documento):
    
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
    
    
    return words, words_tot, median




def save_corpus(vocabulario, dominio):
    
    if dominio == 'entidad':
        with open(os.path.abspath('') + '/generated_data/vocabulario_'+dominio+'.p', 'wb') as handle:
            pickle.dump(vocabulario, handle)
    
    elif dominio == 'categoria':
        with open(os.path.abspath('') + '/generated_data/vocabulario_'+dominio+'.p', 'wb') as handle:
            pickle.dump(vocabulario, handle)
    

def load_corpus(dominio):
    if dominio == 'entidad':
        with open(os.path.abspath('') + '/generated_data/vocabulario_'+dominio+'.p', 'rb') as handle:
            vocabulario = pickle.load(handle)
            
        return vocabulario
    
    elif dominio == 'categoria':
        with open(os.path.abspath('') + '/generated_data/vocabulario_'+dominio+'.p', 'rb') as handle:
            vocabulario = pickle.load(handle)
            
        return vocabulario


def outlayer_detection(df, median, thresh=3.5):
    
    diff  = (df - median)**2
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    
    return modified_z_score > thresh
    

def word_vector(utterances, corpus):
    
    """
    Este es el caso los modelos no secuenciales; se devuelve el vector disperso
    con tantos valores como tenga la bolsa de palabras pero con las frecuencias de las palabras
    que aparezcan en la consulta en cuestion
    """
    
    words = []
    training = []
    
    for text in utterances:
        # Tokenizo cada frase
        w = re.findall(r'\w+', text.lower(),flags = re.UNICODE) # Paso a minusculas todo
        words = w
        # Eliminación de las stop_words 
        words = [word for word in words if word not in stopwords.words('english')]
        # Elimino guiones y otros simbolos raros
        words = [word for word in words if not word.isdigit()] # Elimino numeros    
        # Stemming 
        words = [stemmer.stem(w) for w in words]
        # Pongo como UNK las que no estan en el vocabulario - flattening
        words = [x if x in corpus else 'UNK' for x in words]
        # Inicializo la bolsa de palabras
        bag_new = []
        bag_final = []
    
        # Añado el vector de palabras posibles
        [bag_new.append([w, 0]) for w in corpus]
        
        df_bag = pd.DataFrame(bag_new)
        df_bag.columns = ['palabras', 'ocurrencias']
        
        df_pattern = pd.DataFrame(words)
        df_pattern['ocurrencias'] = 1
        df_pattern.columns = ['palabras', 'ocurrencias']
        df_pattern = df_pattern.groupby(['palabras'])['ocurrencias'].sum() # En este pundo, al pasarlo a indices, se ordenan
        df_pattern = df_pattern.reset_index(level=['palabras'])
        
        df = pd.merge(df_bag, df_pattern, on = 'palabras', how = 'left').fillna(0)
        bag_final = df['ocurrencias_y'].tolist()  
        training.append(bag_final)
        
    # Devuelvo el vector numerico con su TF
    return training


def word2idx_creation(utterances, corpus):
    
    """
    Caso secuencial en el que cada palabra la codifico segun un valor unico asociado
    a la misma en el diccionario
    """
    words = []
    X = []
    
    word2idx = {'START': 0, 'END': 1} # Start/End Tokens. Inicialmente así mi frase es: START END
    current_idx = 2 # El indice comenzará desde la posición 2
    
    for text in utterances:
        # Tokenizo cada frase
        w = re.findall(r'\w+', text.lower(),flags = re.UNICODE) # Paso a minusculas todo
        words = w
        # Eliminación de las stop_words 
        words = [word for word in words if word not in stopwords.words('english')]
        # Elimino guiones y otros simbolos raros
        words = [word for word in words if not word.isdigit()] # Elimino numeros        
        # Stemming 
        words = [stemmer.stem(w) for w in words]
        # Pongo como UNK las que no estan en el vocabulario - flattening
        words = [x if x in corpus else 'UNK' for x in words]
    
        sentence = []
    
        for t in words:
            if t not in word2idx:
                word2idx[t] = current_idx
                current_idx += 1 # El índice lo voy aumentando a medida que añado tokens
            idx = word2idx[t]
            sentence.append(idx)
    
        # Guardo la frase final
        X.append(np.array(sentence)) # lo guardo como un numpy array
        
    return word2idx, X


def word2idx_conversion(utterances, corpus, word2idx):
    words = []
    X = []
    
    for text in utterances:
        # Tokenizo cada frase
        w = re.findall(r'\w+', text.lower(),flags = re.UNICODE) # Paso a minusculas todo 
        words = w
        # Eliminación de las stop_words 
        words = [word for word in words if word not in stopwords.words('english')]
        # Elimino guiones y otros simbolos raros
        words = [word for word in words if not word.isdigit()] # Elimino numeros      
        # Stemming 
        words = [stemmer.stem(w) for w in words]
        # Pongo como UNK las que no estan en el vocabulario - flattening
        words = [x if x in corpus else 'UNK' for x in words]
        
        sentence = []
        
        for t in words:
            if t in word2idx:
                idx = word2idx[t]
            else:
                t = 'UNK'
                idx = word2idx[t]
            sentence.append(idx)
            
        # Guardo la frase final
        X.append(np.array(sentence)) # lo guardo como un numpy array
    
    return X
    
 
def corpus_generation(df, dominio):
    
    vocabulario = {}
    global_vocab = []
    
    if dominio == 'entidad':
        df_domain_total_entity = df
        
        # Compruebo si ya esta creado y lo cargo
        try:
            size1 = len(df_domain_total_entity)
            j = 1
            for domain in list(df_domain_total_entity.keys()):
                k = 1
                for df in df_domain_total_entity[domain]:
                    size2 = len(df_domain_total_entity[domain])
                    entity = list(df.keys())[0]
                    df = list(df.values())[0]

                    print("Iteracion "+str(k)+ " de "+str(size2)+" de " +str(j)+ " de " +str(size1))
                    k += 1
                    
                    vocabulario = load_corpus(dominio = 'entidad')
                j += 1
            
            print("Vocabulario para las entidades cargado")
            
        except:
            print("Generando el vocabulario")
            # df = df_domain_total_entity['automotive'][0]['AB Volvo']
            #titulo = 'AB Volvo'
            
            size1 = len(df_domain_total_entity) 
            j = 1
            for domain in list(df_domain_total_entity.keys()):
                k = 1
                for df in df_domain_total_entity[domain]:
                    size2 = len(df_domain_total_entity[domain])
                    entity = list(df.keys())[0]
                    df = list(df.values())[0]

                    print("Iteracion "+str(k)+ " de "+str(size2)+" de " +str(j)+ " de " +str(size1))
                    k += 1
                    
                    documento = list(df["text"])
                    words, words_tot, median = word_processing(documento)
                    vocabulario[entity] = [words, words_tot, median]
                    global_vocab.append(words)
                    save_corpus(vocabulario, dominio = 'entidad')
                j += 1
                    
            print("Vocabulario para las entidades generado y persistido")
            
        return vocabulario
        
    elif dominio == 'categoria':
        df_domain_total = df
        
        # Compruebo si ya esta creado y lo cargo
        try:
            size1 = len(df_domain_total)
            j = 1
            for df in df_domain_total:
                k = 1
                
                size2 = len(df_domain_total_entity[domain])
                entity = list(df.keys())[0]
                df = list(df.values())[0]

                print("Iteracion "+str(k)+ " de "+str(size2)+" de " +str(j)+ " de " +str(size1))
                k += 1
                
                vocabulario = load_corpus(dominio = 'categoria')
                j += 1
            
            print("Vocabulario para las entidades cargado")
            
        except:
            print("Generando el vocabulario")
            # df = df_domain_total_entity['automotive'][0]['AB Volvo']
            #titulo = 'AB Volvo'
            
            size1 = len(df_domain_total) 
            j = 1
            for df in df_domain_total:
                k = 1
                domain = list(df.keys())[0]
                df = list(df.values())[0]
                size2 = len(df)

                print("Iteracion "+str(k)+ " de "+str(size2)+" de " +str(j)+ " de " +str(size1))
                k += 1
                
                documento = list(df["text"])
                words, words_tot, median = word_processing(documento)
                vocabulario[domain] = [words, words_tot, median]
                global_vocab.append(words)
                save_corpus(vocabulario, dominio = 'categoria')
                j += 1
                    
            print("Vocabulario para las entidades generado y persistido")
            
        return vocabulario
        
    else: # dominio completo
        # ToDo
        print("")
        return ""
    
    
## Creo un df con los datos agregados
#df_final = obtain_train_corpus()
#
## Puedo separarlo en distintos df segun el dominio
#df_domain_total = [{category:df_domain} for category, df_domain in df_final.groupby('category')]
#
## Tambien puedo separar a nivel de dominio y entity
#df_domain_total_entity = {}
#for df in df_domain_total:
#    category = list(df.keys())[0]
#    df = list(df.values())[0]
#    df_entities = [{entity:df_entity} for entity, df_entity in df.groupby('entity_name')]
#    df_domain_total_entity.update({category:df_entities})
#    
#dominio = "entidad"
    
#        
#    vocabulario = corpus_generation(df_domain_total_entity, "entidad")
#    
#    vocabulario_category = corpus_generation(df_domain_total, "categoria")
