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
    

    ######################### Test ###########################################
    dominio = "entidad"
    d_resultados = {}

    df_final_test = obtain_test_corpus()
    
    # Puedo separarlo en distintos df segun el dominio
    df_domain_total_test = [{category:df_domain} for category, df_domain in df_final_test.groupby('category')]
    
    # Tambien puedo separar a nivel de dominio y entity
    df_domain_total_entity_test = {}
    for df in df_domain_total_test:
        category = list(df.keys())[0]
        df = list(df.values())[0]
        df_entities = [{entity:df_entity} for entity, df_entity in df.groupby('entity_name')]
        df_domain_total_entity_test.update({category:df_entities})
    
    # Cargo el vocabulario generado
    vocabulario = load_corpus(dominio)
    entidades = list(vocabulario.keys())
    categorias = list(df_domain_total_entity_test.keys())
    
    
    i = 1
    total = len(entidades)
    for categoria in categorias:
        for df in df_domain_total_entity_test[categoria]:
            print("Clasificación " +  str(i) + "/" + str(total))

            entidad = list(df.keys())[0]
            df = list(df.values())[0]
            corpus = vocabulario[entidad][0]
            
            ###### n-gram modeling - trigram
            d = {}
            sentences = list(df["text"])
            sentences = [word_preparing_test([documento], corpus) for documento in sentences]
            ngram_type =  "trigram"
            
            # Cargo el n-gram model
            # Eliminio los backslashes de las palabras que los tengan
            if '/' in entidad:
                    entidad= entidad.replace('/', '_')
                        
            with open(os.path.abspath('') + r'/generated_data/_model_n_gram_'+str(entidad)+'_'+ ngram_type +'.p', 'rb') as handle:
                    model = pickle.load(handle)
            
            model_test = model_generation(sentences, ngram_type)
    
            # Evaluacion con C-E
            ngram_test = list(model_test.freqdist().keys()) # Saco los ngramas que hay
            c_e = evaluacion_c_e(model, model_test, ngram_test)
            
            ngram_train = list(model.freqdist().keys())
            e = evaluacion_e(model, ngram_train)
            
            d['dominio'] = dominio
            d['ngram_type'] = ngram_type
            d['categoria'] = categoria
            d['entidad'] = entidad
            d['c-e'] = c_e
            d['e'] = e
            
            var = 'ngram_model' + '_' + ngram_type + '_' + categoria + '_' + entidad
            d_resultados[var] = d
            
            i+=1

            ###### n-gram modeling - bigram
    i = 1
    total = len(entidades)
    for categoria in categorias:
        for df in df_domain_total_entity_test[categoria]: 
            entidad = list(df.keys())[0]
            df = list(df.values())[0]
            corpus = vocabulario[entidad][0]
            print("Clasificación " +  str(i) + "/" + str(total))
        
            d = {}
            ngram_type =  "bigram"
            
            # Cargo el n-gram model
            # Eliminio los backslashes de las palabras que los tengan
            if '/' in entidad:
                    entidad= entidad.replace('/', '_')
            
            with open(os.path.abspath('') + r'/generated_data/_model_n_gram_'+str(entidad)+'_'+ ngram_type +'.p', 'rb') as handle:
                    model = pickle.load(handle)      
                    
            model_test = model_generation(sentences, ngram_type)
            
            # Evaluacion con C-E
            ngram_test = list(model_test.freqdist().keys()) # Saco los ngramas que hay
            c_e = evaluacion_c_e(model, model_test, ngram_test)
            
            ngram_train = list(model.freqdist().keys())
            e = evaluacion_e(model, ngram_train)
            
            d['dominio'] = dominio
            d['ngram_type'] = ngram_type
            d['categoria'] = categoria
            d['entidad'] = entidad
            d['c-e'] = c_e
            d['e'] = e
            
            var = 'ngram_model' + '_' + ngram_type + '_' + categoria + '_' + entidad
            d_resultados[var] = d
            
            i += 1
            
            ##### Polarity Test - RF  
    
    d_resultados_2 = {}
    
    i = 1
    total = len(entidades)
    for categoria in categorias:
        for df in df_domain_total_entity_test[categoria]: 
            entidad = list(df.keys())[0]
            df = list(df.values())[0]
            corpus = vocabulario[entidad][0]
            print("Clasificación " +  str(i) + "/" + str(total))
            
            d = {}
            
            # Eliminio los backslashes de las palabras que los tengan
            if '/' in entidad:
                    entidad= entidad.replace('/', '_')
                    
            with open(os.path.abspath('') + r'/generated_data/model_rf_polarity_'+str(entidad)+'.p', 'rb') as handle:
                classifier = pickle.load(handle)
                
            # Cargo los encodings
            with open(os.path.abspath('') + r'/generated_data/encodings.p', 'rb') as handle:
                encodings = pickle.load(handle)
            polarity = encodings['polarity']
            
            X = list(df['text'])
            Xt = word_vector(X, corpus)
            X = Xt
            
            y = df['polarity']
            y = np.array([polarity[j] for j in y])
            
            cm, accuracy, average_precision, recall, f1 = model_test_eval(corpus, classifier, X, y, scaling=False)
            
            d['dominio'] = dominio
            d['algoritmo'] = 'random_forest'
            d['categoria'] = categoria
            d['entidad'] = entidad
            d['cm'] = cm
            d['accuracy'] = accuracy
            d['average_precision'] = average_precision
            d['recall'] = recall
            d['f1'] = f1
            d['caso'] = "polarity"
            
            var = 'polarity' + '_' + 'random_forest' + '_' + categoria + '_' + entidad
            d_resultados_2[var] = d            
            
            i += 1
            
            ##### Polarity Test - CNN+RNN
    i = 1
    total = len(entidades)
    for categoria in categorias:
        for df in df_domain_total_entity_test[categoria]: 
            entidad = list(df.keys())[0]
            df = list(df.values())[0]
            corpus = vocabulario[entidad][0]
            print("Clasificación " +  str(i) + "/" + str(total))
            
            d = {}
            
            # Eliminio los backslashes de las palabras que los tengan
            if '/' in entidad:
                    entidad= entidad.replace('/', '_')
            
            model = load_model(os.path.abspath('') + "/generated_data/cnn_keras_model_polarity_"+str(entidad)+".h5")
            
            cm, accuracy, average_precision, recall, f1 =  model_test_eval_rnn(model, df, corpus, entidad)
            
            d['dominio'] = dominio
            d['algoritmo'] = 'cnn_rnn'
            d['categoria'] = categoria
            d['entidad'] = entidad
            d['cm'] = cm
            d['accuracy'] = accuracy
            d['average_precision'] = average_precision
            d['recall'] = recall
            d['f1'] = f1
            d['caso'] = "polarity"
               
            var = 'polarity' + '_' + 'cnn_rnn' + '_' + categoria + '_' + entidad
            d_resultados_2[var] = d 
            
            i += 1
            
            
            ##### Topic Detection - RF
    i = 1
    total = len(entidades)
    for categoria in categorias:
        for df in df_domain_total_entity_test[categoria]: 
            entidad = list(df.keys())[0]
            df = list(df.values())[0]
            corpus = vocabulario[entidad][0]
            print("Clasificación " +  str(i) + "/" + str(total))            
            
            
            d = {}
            
            # Eliminio los backslashes de las palabras que los tengan
            if '/' in entidad:
                    entidad= entidad.replace('/', '_')
            
            with open(os.path.abspath('') + r'/generated_data/model_rf_topic_detection_'+str(entidad)+'.p', 'rb') as handle:
                classifier = pickle.load(handle)
         
            
#            X = list(df['text'])
#            Xt = word_vector(X, corpus)
#            X = Xt
            
            # Cargo los encodings
            with open(os.path.abspath('') + r'/generated_data/encodings.p', 'rb') as handle:
                encodings = pickle.load(handle)
            topic = encodings['topic']
            
            # Filtro los topics que no estan en train
            topic_l = list(topic.keys())
            df_f = df.copy()
            df_f = df_f[df_f['topic'].isin(topic_l)]
            
            # En caso de estar vacío por no coincidir las categorias, salto a la siguiente iteracion
            if len(df_f) == 0:
                d['dominio'] = dominio
                d['algoritmo'] = 'random_forest'
                d['categoria'] = categoria
                d['entidad'] = entidad
                d['cm'] = None
                d['accuracy'] = None
                d['average_precision'] = None
                d['recall'] = None
                d['f1'] = None
                d['caso'] = "topic_detection"
                   
                var = 'topic_detection' + '_' + 'random_forest' + '_' + categoria + '_' + entidad
                d_resultados_2[var] = d 
                
                continue
            
            
            X = list(df_f['text'])
            Xt = word_vector(X, corpus)
            X = Xt
            
            y = df_f['topic']
            y = np.array([topic[j] for j in y])
            
   
#            y = list(df['topic'])
#            labelencoder_X = LabelEncoder()
#            y=labelencoder_X.fit_transform(y) # Codifico en valores numericos las clases que hay
#            y_original = y
#            
#            if max(y_original) != 1 and sum(y_original) != 0:
#                    # Encoding a one-hot
#                    y = y.reshape(-1, 1)
#                    onehotencoder = OneHotEncoder()
#                    y = onehotencoder.fit_transform(y).toarray()
#                    
#                    # Remove dummy variable trap
#                    y = y[:, 1:] # Elimino una de las columnas por ser linearmente dependiente de las demas
            
            
            cm, accuracy, average_precision, recall, f1 = model_test_eval(corpus, classifier, X, y, scaling=False)
                        
            d['dominio'] = dominio
            d['algoritmo'] = 'random_forest'
            d['categoria'] = categoria
            d['entidad'] = entidad
            d['cm'] = cm
            d['accuracy'] = accuracy
            d['average_precision'] = average_precision
            d['recall'] = recall
            d['f1'] = f1
            d['caso'] = "topic_detection"
               
            var = 'topic_detection' + '_' + 'random_forest' + '_' + categoria + '_' + entidad
            d_resultados_2[var] = d       
            
            i += 1
        
        
            ##### Topic Detection - TF IDF
    i = 1
    total = len(entidades)
    for categoria in categorias:
        for df in df_domain_total_entity_test[categoria]: 
            entidad = list(df.keys())[0]
            df = list(df.values())[0]
            corpus = vocabulario[entidad][0]
            print("Clasificación " +  str(i) + "/" + str(total))            
            
            
            d = {}
            
            # Eliminio los backslashes de las palabras que los tengan
            if '/' in entidad:
                    entidad= entidad.replace('/', '_')
            
            with open(os.path.abspath('') + r'/generated_data/model_tf_idf_topic_detection_'+str(entidad)+'.p', 'rb') as handle:
                df_classificacion = pickle.load(handle)
                
            # Cargo los encodings
            with open(os.path.abspath('') + r'/generated_data/encodings.p', 'rb') as handle:
                    encodings = pickle.load(handle)
            topic = encodings['topic']    
            
            
            X_tf_idf = list(df_classificacion['tf-idf'])
            y_tf_idf = list(df_classificacion['topic'])
            
            
            # Encoding a numerico
            y_tf_idf = np.array([topic[i] for i in y_tf_idf])
            
            X_train = X_tf_idf
            y_train = y_tf_idf
#            y_train = np.array([topic[i] for i in y_train])
            
            
            # Filtro los topics que no estan en train
            topic_l = list(topic.keys())
            df_f = df.copy()
            df_f = df_f[df_f['topic'].isin(topic_l)]
            
            # En caso de estar vacío por no coincidir las categorias, salto a la siguiente iteracion
            if len(df_f) == 0:
                d['dominio'] = dominio
                d['algoritmo'] = 'random_forest'
                d['categoria'] = categoria
                d['entidad'] = entidad
                d['cm'] = None
                d['accuracy'] = None
                d['average_precision'] = None
                d['recall'] = None
                d['f1'] = None
                d['caso'] = "topic_detection"
                   
                var = 'topic_detection' + '_' + 'random_forest' + '_' + categoria + '_' + entidad
                d_resultados_2[var] = d 
                
                continue
 
            X = list(df_f['text'])
            
#            Xt = word_vector(X, corpus)
#            X = Xt
            
#            y = df_f['topic']
#            y = np.array([topic[j] for j in y])
            
            
#            X = list(df['text'])
            
            words, words_tot, median, df_pattern, df_suma = word_tf_idf(X)
            
            df_f.reset_index(inplace=True)
            df_classificacion = df_suma.join(df_f, how="outer") # Join por los index
            

            X_val = list(df_classificacion['tf-idf'])
            # Encoding a numerico
            y_val = list(df_classificacion['topic'])
            y_val = np.array([topic[i] for i in y_val])
            
#            y_val = list(df_classificacion['topic'])
#            y_val = labelencoder_X.fit_transform(y_val)
            
     
            # Menor distancia cuadratica de TF
            y_pred = []
            for x_ref in X_val:
                ref = 999
                i = 0
                for x in X_train:
                    
                    diff  = (x_ref - x)**2
                    diff = np.sqrt(diff)
                    print(diff)
                    
                    if diff < ref:
                        i = X_train.index(x)
                        ref = diff

                y_pred.append(y_train[i])  # Identifico con la clase de menor distancia cuadratica TF-IDF
                

            # Deshago cambios
            y_val = pd.DataFrame(y_val)
            y_val = [(np.argmax(np.asarray(x)) + 1.0) if max(np.asarray(x)) > 0 else 0.0 for x in y_val.values.tolist()]
            
            y_pred = [(np.argmax(np.asarray(x)) + 1.0) if max(np.asarray(x)) > 0 else 0.0 for x in y_pred]           

    
            cm = confusion_matrix(y_val, y_pred)
            
            # Accuracy
            accuracy = accuracy_score(y_val, y_pred)
            
            # Precision
            average_precision = precision_score(y_val, y_pred, average = "macro")
            
            # Recall
            recall = recall_score(y_val, y_pred, average='macro')
            
            # F1
            f1 = f1_score(y_val, y_pred, average='weighted', labels=np.unique(y_pred))

            d['dominio'] = dominio
            d['algoritmo'] = 'tf_idf'
            d['categoria'] = categoria
            d['entidad'] = entidad
            d['cm'] = cm
            d['accuracy'] = accuracy
            d['average_precision'] = average_precision
            d['recall'] = recall
            d['f1'] = f1
            d['caso'] = "topic_detection"
            
            var = 'topic_detection' + '_' + 'tf_idf' + '_' + categoria + '_' + entidad
            d_resultados_2[var] = d              
            
            i += 1
 
            ##### Priority Detection - RF
    i = 1
    total = len(entidades)
    for categoria in categorias:
        for df in df_domain_total_entity_test[categoria]: 
            entidad = list(df.keys())[0]
            df = list(df.values())[0]
            corpus = vocabulario[entidad][0]
            print("Clasificación " +  str(i) + "/" + str(total))            
            
            
            d = {}
            
            # Eliminio los backslashes de las palabras que los tengan
            if '/' in entidad:
                    entidad= entidad.replace('/', '_')
            
            with open(os.path.abspath('') + r'/generated_data/model_rf_priority_detection_'+str(entidad)+'.p', 'rb') as handle:
                classifier = pickle.load(handle)
         
            # Cargo los encodings
            with open(os.path.abspath('') + r'/generated_data/encodings.p', 'rb') as handle:
                    encodings = pickle.load(handle)
            topic_priority = encodings['topic_priority']

            X = list(df['text'])
            Xt = word_vector(X, corpus)
            X = Xt
            
            y = df['topic_priority']
            y = np.array([topic_priority[j] for j in y])
            
            cm, accuracy, average_precision, recall, f1 = model_test_eval(corpus, classifier, X, y, scaling=False)
            
            d['dominio'] = dominio
            d['algoritmo'] = 'random_forest'
            d['categoria'] = categoria
            d['entidad'] = entidad
            d['cm'] = cm
            d['accuracy'] = accuracy
            d['average_precision'] = average_precision
            d['recall'] = recall
            d['f1'] = f1
            d['caso'] = "priority"
            
            var = 'priority' + '_' + 'random_forest' + '_' + categoria + '_' + entidad
            d_resultados_2[var] = d         
        
            i += 1
            
 #######################################################################################

