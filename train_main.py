# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 17:51:14 2018

@author: alber
"""


from data_preprocessing import *
from model_training_polarity import *
from model_training_priority import *
from model_training_topic_detection import *
from ngram_modeling import *
from models_test import *

if __name__ == "__main__":
    
    
    ######################### Train ###########################################
    
    # Creo un df con los datos agregados
    df_final = obtain_train_corpus()
    
    # Puedo separarlo en distintos df segun el dominio
    df_domain_total = [{category:df_domain} for category, df_domain in df_final.groupby('category')]
    
    # Tambien puedo separar a nivel de dominio y entity
    df_domain_total_entity = {}
    for df in df_domain_total:
        category = list(df.keys())[0]
        df = list(df.values())[0]
        df_entities = [{entity:df_entity} for entity, df_entity in df.groupby('entity_name')]
        df_domain_total_entity.update({category:df_entities})
      
        
    # Entreno a nivel de entidad
    dominio = "entidad"
    
    ########## Polarity Train #################
    # Entrenamiento del modelo de RF - polarity
    rf_polarity_train(dominio)
    
    # Entrenamiento del modelo de CNN/RNN - polarity
    cnn_rnn_polarity_train(dominio)

    ########## Priority Train ################# 
    # Entrenamiento del modelo RF - priority
    rf_priority_train(df, dominio)
    
    ########## Topic Detection Train #################
    # Entrenamiento del modelo de RF - topic detection
    rf_topic_train(df, dominio)
    
    # Entrenamiento del modelo de TF-IDF con MAE - topic detection
    tf_idf_train(df, dominio)


    ########## n-gram generation #################
    ngram_type =  "trigram"
    create_models(dominio, ngram_type)
    
    ngram_type =  "bigram"
    create_models(dominio, ngram_type)
    
    
