# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:33:05 2018

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

from nltk import bigrams, trigrams
from collections import Counter, defaultdict
import random
import nltk
from nltk.util import ngrams
import pandas as pd

from nltk import probability
from nltk.probability import FreqDist, WittenBellProbDist, KneserNeyProbDist, SimpleGoodTuringProbDist, sum_logs, entropy
from nltk.probability import log_likelihood
import math

from data_preprocessing import word_vector, corpus_generation, obtain_train_corpus, word2idx_creation, word2idx_conversion
global stemmer
stemmer = SnowballStemmer("english")

def word_preparing(documento):
    
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

    return word_list




def model_generation(sentences, ngram_type):
    
    if ngram_type == 'bigram':
        bigram_model =  [ ngram for sent in sentences 
            for ngram in ngrams(sent, 2, pad_left = True, pad_right = True, right_pad_symbol='EOS', left_pad_symbol="BOS")]
        
        fdist_b = FreqDist(bigram_model)
        good_turing_bigram = SimpleGoodTuringProbDist(fdist_b)
        
        return good_turing_bigram
        
        
    elif ngram_type == 'trigram':
        trigram_model = [ ngram for sent in sentences 
            for ngram in ngrams(sent, 3, pad_left = True, pad_right = True, right_pad_symbol='EOS', left_pad_symbol="BOS")]
        
        fdist_t = FreqDist(trigram_model)
        good_turing_trigram = SimpleGoodTuringProbDist(fdist_t)
        
        return good_turing_trigram
        
    

def evaluacion_c_e(ngram_train_prob, ngram_test_prob, ngram_test):
    # Se comprueban los ngramas nuevos y se calcula la cross-entropy para ellos
    # Modificacion sobre la funcion log_likelihood de NLTK
    return -sum(ngram_train_prob.prob(s) * math.log(ngram_test_prob.prob(s), 2)
               for s in ngram_test)
    

def evaluacion_e(ngram_test_prob, ngram_test):
    # Se comprueban los ngramas nuevos y se calcula la cross-entropy para ellos
    # Modificacion sobre la funcion log_likelihood de NLTK
    return -sum(ngram_test_prob.prob(s) * math.log(ngram_test_prob.prob(s), 2)
               for s in ngram_test)
    
    
def create_models(dominio, ngram_type):
    
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
                
                entidad = list(df.keys())[0]
                df = list(df.values())[0]
                corpus = vocabulario[entidad][0]
                
                sentences = list(df["text"])
                sentences = [word_preparing([documento]) for documento in sentences]
                
                #ngram_type = "trigram"
                model = model_generation(sentences, ngram_type)
                
                # Eliminio los backslashes de las palabras que los tengan
                if '/' in entidad:
                        entidad= entidad.replace('/', '_')
            
                 # Persistencia del modelo entrenado
                with open(os.path.abspath('') + r'/generated_data/_model_n_gram_'+str(entidad)+'_'+ ngram_type +'.p', 'wb') as handle:
                        pickle.dump(model, handle)
                        
                print("Modelo "+ str(i)+" entrenado y guardado")
                i += 1
    
    
    
