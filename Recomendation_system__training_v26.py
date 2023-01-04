# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 09:53:03 2022


This application take a text file and extract sentences from that
then cluster the sentense based on their centroied word2vec vector presntation


@author: Hojat
"""
# conda install -c conda-forge scikit-learn-extra
# installing the extra
#pip instal -u mittens
from   mittens import  Mittens
import os
from   sklearn.feature_extraction.text import CountVectorizer
import pickle
import pandas as pd
import hashlib
from   datetime import datetime
from   Carlo_ngrams_tool.utilities_recommendation import \
       extract_text_from_folders_of_pdfs_zip,\
       creat_bench, most_similar,glove2dict
#Define if the program wants to train the model(True) or
# evaluate the input docuements
#Selected mode for training or analysing 'GloVe' or 'word2vec'
model_select='GloVe'
#model file for glove
trained_model='glove_dictionary_test.pkl'

#A pretrained model for starting point
# get it from https://nlp.stanford.edu/projects/glove
pretrained_model="glove_6B_300d.txt"   
 
#################  INPUT FILES
# Defining the path to the trainig pdf files
folder_path = '../training'
#'C:/temp/training.zip' #'../training'

# benchmarks file
file_bench = 'rooted_bench.csv'
# stop words file
stp_file = '../stopwords_en.txt'



######### CONFIGURATION FACTORS
#this file contiend processed file for trainig and use for not 
#process already used files
hashed_trained_file="hash_trained.csv"
# maximum number of fils that has been read for training
max_num_files = 1000
#minimum  accepteable word length
ngram_min    = 2  # minimum ngraming length
ngram_max    = 3  # maximum ngraming length

# maximum # of cluster to estimate the document clustring apply for Elbow and wcss
max_cluster = 25 #50
# if the multiprocessing would be used
MultiProcess=True

# the text length would be exponentiation to (-base_frequency) to calculate minimum
# frequency of the ngrams to be considered.
# the higher value cause the less bigrams extraction
bigram_base_frequency = 0.28
# the higher value cause the less trigrams extraction
trigram_base_frequency = 0.28

# The direct_bench >= this treshhold for selecting  sentences that are siimlar to benchmark
similarity_tershold = .85 #
#a maximum number of words in an input file to split in training
split_value=5000 
# a value which is used for saving trained model with each number of input file
update_period=20

"""
-----main
"""

"""
global paramters setting 
"""
    # manager = mp.Manager()
    # bench_pd_shared = manager.list()
    # word_dict_shared = manager.dict()    
if __name__ == "__main__":
    


#initilization
    stopwords_file = open(stp_file, 'r', encoding='utf8')
    
    # initializing list of stopwords
    stopwords1 = []
    
    # populating the list of stopwords
    for word in stopwords_file:
        stopwords1.append(word.strip())
    
    print('++++++++++++++++++++++++++++++\n')
    print('---Starting preprocessing-')
    start_time = datetime.now()
    print('---Starting time: {}'.format(start_time))

    #This part try to create glove model by using input text files and trian a pretrained GloVe model
    #It will be execute ony one time for creating the glove model
    
    """
    Glove stanford package is out of date and has problem for instaling on python version 3.8
    Then I decided to use Mittens as alternative that works well. the problem  describes here:
    https://stackoverflow.com/questions/60253605/glove-for-python-3-7-version
       
    """

    bench, bench_w ,bench_type_list= creat_bench(file_bench,Training=True)       
    # a list of splited pharses in benchmark
    bench_list=[item.split('_') for item in bench]    
    
    #check if the hash file is already created. 
    #this file contains input file name and a hash of their content
    #after using for triaing. If a file conten is already processed
    #it is ignored for training.
    if(os.path.isfile(hashed_trained_file)):
        df_hash=pd.read_csv(hashed_trained_file)
    else:
        df_hash=pd.DataFrame(columns=["fname","hash","trained_words","new_words"])        
    
    
    trained_until=0 #this variable shows if the trained terminated somewhere and we have trained model up to that point. zero elsewhere
    #a pretrianed model is used for start point
    glove_filename = pretrained_model 
    print("Reading the pretrained Glove vocaboulary .....")
    if(trained_until==0):
        updated_pre_glove = glove2dict(glove_filename)
    else:
        with open(trained_model, 'rb') as f:
            updated_pre_glove = pickle.load(f)    
    pre_similarity=most_similar("contract",updated_pre_glove,10)
    print(pre_similarity)
    start_time = datetime.now()
    print('--- Starting time for training: {}'.format(start_time))
    index=0
    # Large input files make very large coocurrance matrix that slow down 
    #the process exponentially to mitigate the probelm
    # larger file should be split to smaller ones to process 
 
    # split_value set a tershhold for the input file size and the 
    #process split them to smaller if it pass the treshhold

    #iterrate over input document files
    for fname, _, txt, _, _,_, _ in extract_text_from_folders_of_pdfs_zip(bench_list,folder_path,
                 stopwords1 ,max_files=max_num_files):
        # Assumes the default UTF-8
        try:
            hash_object = str(hashlib.md5(txt.encode()).hexdigest())
            if(hash_object in list(df_hash['hash'])):
                print("Ignoring an already applied file:",fname)
                continue
            index+=1            
            txt=txt.split()
            #After every 50 files processing the trained model is saved 
            if(index%update_period==0):
                with open(trained_model, 'wb') as f:
                    pickle.dump(updated_pre_glove, f)
                f.close() 
                df_hash.to_csv(hashed_trained_file,index=False)
        # calculate partision size for larg files
            dividen=int(len(txt)/split_value)+1
            new_added_words_t=0
            total_traiend_len_t=0
            #itterate ove each partion of larg file and use that for training
            for k in range(dividen):
                if k==dividen-1:
                    doc=txt[int(k*len(txt)/dividen):]
                else:             
                    doc=txt[int(k*len(txt)/dividen):int((k+1)*len(txt)/dividen)]
                #find input doc unique set of words
                corp_vocab = list(set(doc))
    # use the practices for traiing GloVe model from:
    #https://towardsdatascience.com/fine-tune-glove-embeddings-using-mittens-89b5f3fe4c39
                
                cv = CountVectorizer(ngram_range=(1,1), vocabulary=corp_vocab)
                X = cv.fit_transform(corp_vocab)
                Xc = (X.T * X)
                Xc.setdiag(0)
                coocc_ar = Xc.toarray() 
                new_added_words=len(set(corp_vocab)-set(list(updated_pre_glove.keys())))
                total_trainang_len=len(set(corp_vocab))
                new_added_words_t+=new_added_words
                total_traiend_len_t+=total_trainang_len                
                print("<<%s>>Adding to GloVe(part=%d(%d)/total=%d/unique=%d/new=%d)"%(fname,k,dividen,len(doc),
                    total_trainang_len,new_added_words))
                mittens_model = Mittens(n=300, max_iter=100)
                
                new_embeddings = mittens_model.fit(
                    coocc_ar,
                    vocab=corp_vocab,
                    initial_embedding_dict= updated_pre_glove)
                
                new_embed_dic={k:v for k,v in zip(corp_vocab,new_embeddings)}
            #https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression
                updated_pre_glove = {**updated_pre_glove, **new_embed_dic}
               #update hash file
                new_row = pd.DataFrame({'fname':fname, 'hash':hash_object, 
                                          'trained_words':total_trainang_len,
                                          'new_words':new_added_words_t}, index=[0])
                df_hash = pd.concat([new_row,df_hash.loc[:]]).reset_index(drop=True)        
            # df_hash = df_hash.append({'fname':fname, 'hash':hash_object, 
            #                           'trained_words':total_trainang_len,
            #                           'new_words':new_added_words_t},
            #                           ignore_index=True)            
        
        except Exception as e:
            print("an error occured and program terminated:",e)
    print("---------------------------------------------------")        
    with open(trained_model, 'wb') as f:
        pickle.dump(updated_pre_glove, f)
    print('--- Total training duration: {}'.format(datetime.now() - start_time))

    print("---------------------------------------------------")        
    with open(trained_model, 'rb') as f:
        updated_pre_glove = pickle.load(f)
    print(most_similar("contract",updated_pre_glove,10))
    word_dict=updated_pre_glove
    vocab=word_dict.keys()
 
