# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:50:50 2018

@author: alber
"""
#### Librerias

from data_preprocessing import *
from model_training_polarity import *
from model_training_priority import *
from model_training_topic_detection import *

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


#### Data preparation y carga de datos

def word_preparing(documento):
    
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

    
    
    return word_list



dominio = "entidad"

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


corpus = vocabulario[entidad][0]


#### Implementacion manual de algoritmos con DataFrames

# Sin Smoothing
bigram_model =  [ ngram for sent in sentences 
            for ngram in ngrams(sent, 2, pad_left = True, pad_right = True, right_pad_symbol='EOS', left_pad_symbol="BOS")]

freq_dist = nltk.FreqDist(bigram_model) # Estas freq son los BIGRAM COUNTS, no las frecuencias relativas (falta dividirlas por el numero de bigrams de igual palabra de inicio)



# Veces de tokens de inicio (c(wx))
dict_c_wx = {}
freq_dist_l = list(freq_dist.items())
for tup in freq_dist_l:
    try: 
        # Da key error si no existe aun 
        dict_c_wx[tup[0][0]] += tup[1]
    except:
        dict_c_wx[tup[0][0]] = tup[1]


# Bigrams distintos por palabra de inicio (T(wx)) - cuento las veces que aparece uno de esos bigramas
dict_t_wx = {}
freq_dist_l = list(freq_dist.items())
for tup in freq_dist_l:
    try: 
        # Da key error si no existe aun 
        dict_t_wx[tup[0][0]] += 1
    except:
        dict_t_wx[tup[0][0]] = 1



corpus = list(set(corpus))


### One smoothing
df_freq = pd.DataFrame()

corpus_in = corpus + ['BOS']
df_freq["palabras"] = corpus_in
df_freq.set_index('palabras', inplace=True)
df_freq["palabras"] = corpus_in
corpus_end = corpus + ['EOS']


v = len(corpus_in)
# Creo las columnas
for w2 in corpus_end:
    
    # count
    # df_freq[w2] = 1
    # probabilities
    df_freq[w2] = 1/v
    

## Pongo los datos
for w2 in corpus_end:
    for w1 in corpus_in:
        if (w1,w2) in freq_dist:
            # counts
#            df_freq.at[w2,w1] = freq_dist[(w1,w2)] + 1 
            
            # probabilities
            c_wn_1_w_n = freq_dist[(w1,w2)]
            c_wn_1 = dict_c_wx[w1]
            df_freq.at[w2,w1] = (c_wn_1_w_n + 1) / (c_wn_1 + v)


### Written-Bell
# Aqui a los n-gramas que no hemos visto los modelizamos con la probabilidad de ver un n-grama por primera vez
# Lo saco con el numero de veces que veo un n-grama por primera vez en training = nº de n-gramas distintos
# Y para cada bigama wn-1,wn-2 no contemplado saco su probabilidad atendiendo al historico y a las veces que se ha
# dado un bigram que empiece por wn-1

t = len(list(freq_dist.items()))
n = len(corpus)


df_freq_wb = pd.DataFrame()

df_freq_wb["palabras"] = corpus_in
df_freq_wb.set_index('palabras', inplace=True)
df_freq_wb["palabras"] = corpus_in

# Para los bigramas que empiezan por una palabra que no esta en el historico como primera palabra, lo trato como unigramas
z = (len(df_freq_wb) * len(df_freq_wb)) - t # n-gramas no vistos, los totales (datos de la matriz) menos los vistos
p_wb = t / ((n + t) * z)

# Creo las columnas
for w2 in corpus_end:
    df_freq_wb[w2] = p_wb
    


## Para los bigrams que tienen como primera palabra una de las que aparece en los bigrams de entrenamiento, pero ese bigram no está
n = len(corpus_in)
for w1 in corpus_in:
    if w1 in dict_t_wx:
        t_wi = dict_t_wx[w1]
        z_wi = n - t_wi      
        df_freq_wb.loc[w1] =  t_wi / (z_wi * (n + t_wi)) # pongo la probabilidad para ese caso
        

### Para los bigrams que están
for w2 in corpus_end:
    for w1 in corpus_in:
        if (w1,w2) in freq_dist:
            c_wx_wi = freq_dist[(w1,w2)]
            c_wx = dict_c_wx[w1]
            t_wx = dict_t_wx[w1]
            z_wx = n - t_wx
            df_freq_wb.at[w2,w1] = (1/z_wx) * (c_wx_wi / (c_wx + t_wx))


#### Implementacion con NLTK
bigram_model =  [ ngram for sent in sentences 
            for ngram in ngrams(sent, 2, pad_left = True, pad_right = True, right_pad_symbol='EOS', left_pad_symbol="BOS")]

trigram_model = [ ngram for sent in sentences 
            for ngram in ngrams(sent, 3, pad_left = True, pad_right = True, right_pad_symbol='EOS', left_pad_symbol="BOS")]

fdist_b = FreqDist(bigram_model)
fdist_t = FreqDist(trigram_model)

aa = WittenBellProbDist(fdist_t)

good_turing_trigram = SimpleGoodTuringProbDist(fdist_t)

kneser_trigram = KneserNeyProbDist(fdist_t, bins = None, discount=0.75)


#### Evaluacion
prob_sum = 0
for i in kneser_trigram.samples():
    if i[0] == "I" and i[1] == "confess":
        prob_sum += kneser_trigram.prob(i)
        print ("{0}:{1}".format(i, kneser_trigram.prob(i)))
print(prob_sum)


e = entropy(kneser_trigram)

# Para la evaluacion se saca la distribucion de probabilidad del tezt (sus trigrams, bigrams) y se compara con la de train
prueba = ['hola', 'dos', 'perro', 'gato', 'http', 'perros', 'ly', 'offer', 'free', 'onlin']*2
trigram_prueba = ngrams(prueba, 3, pad_left = True, pad_right = True, right_pad_symbol='EOS', left_pad_symbol="BOS")
fdist_prueba = FreqDist(trigram_prueba) # saco las frecuencias
kneser_prueba = KneserNeyProbDist(fdist_prueba, bins = None, discount=0.75) # saco la dist de prob

evaluacion = log_likelihood(kneser_trigram, kneser_trigram)


probs = (kneser_trigram.prob(s) for s in kneser_trigram.samples())
z = -sum(p * math.log(p,2) for p in probs)


a = [kneser_trigram.prob(s) * math.log(kneser_prueba.prob(s), 2) for s in trigram_prueba]


def evaluacion_c_e(ngram_train_prob, ngram_test_prob, ngram_test):
    # Se comprueban los ngramas nuevos y se calcula la cross-entropy para ellos
    # Modificacion sobre la funcion log_likelihood de NLTK
    return sum(ngram_train_prob.prob(s) * math.log(ngram_test_prob.prob(s), 2)
               for s in ngram_test)



gt = SimpleGoodTuringProbDist(fdist_prueba) # saco la dist de prob
a = [good_turing_trigram.prob(s) * math.log(gt.prob(s), 2) for s in trigram_prueba]

