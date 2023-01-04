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
import numpy as np
#normalize a vector
import pickle
import pandas as pd

import time
from datetime import datetime
from Carlo_ngrams_tool.utilities_recommendation import \
     bench_clustring,\
   extract_text_from_folders_of_pdfs_zip,\
   creat_bench,normalise,\
   get_vector,find_sentences,words_score2,evaluate_NER
#import multiprocessing 
#from pyemd import emd
#conda install -c conda-forge geocoder
#https://geocoder.readthedocs.io/api.html#forward-geocoding

#Define if the program wants to train the model(True) or
# evaluate the input docuements
Training=False
#model_select='word2vec'
#Selected mode for training or analysing 'GloVe' or 'word2vec'
model_select='GloVe'
#model file for glove
trained_model='glove_dictionary_test.pkl'

#A pretrained model for starting point
# get it from https://nlp.stanford.edu/projects/glove
pretrained_model="glove_6B_300d.txt"   

#################  INPUT FILES
# Defining the path to the trainig pdf files
folder_path = '../training'#'C:/temp/training.zip' #'../training'
# folders for input files to be evaluated
#proj_dir = '../Manually Identified Documents'
proj_dir = '../Manually Identified Documents'
# benchmarks file
file_bench = 'rooted_benchmarks_11_07_2022.csv'
# stop words file
stp_file = '../stopwords_en.txt'


################# OUTPUT FILES
#Presenting the similarity degree for each file in proj_dir to the entire benchmark
sim_to_bench_file='sim_to_bench.csv'

######### CONFIGURATION FACTORS

# maximum number of fils that has been read for training
max_num_files = 1000

# maximum # of cluster to estimate the document clustring apply for Elbow and wcss
max_cluster = 25 #50

#required Name Entitiy Recognition (NER) in text preprocessing
NER=True

#input file from Crawler
crawler_file='bench-to-crawled-duck_unique.csv'





with open('glove_dictionary.pkl', 'rb') as f:
    word_dict = pickle.load(f)
vocab=word_dict.keys()    
bench, bench_w ,bench_type_list= creat_bench(file_bench,Training=Training,vocab=vocab)           
bench_list=[item.split('_') for item in bench]    
     
        # creat a df from the benchmarks and their weights
bench_df = pd.DataFrame({'benchmark': bench, 'weight': bench_w,'type':bench_type_list})

#add catagory to benchmakr dataframe
bench_clustring(bench_df,word_dict,max_cluster=50)
print("++++++++++++++++++++++++++++++++++++++++++\n")
bench_df=bench_df.sort_values(by=["class"]).reset_index(drop=True)

bench_df['weight_int']=(bench_df.weight/bench_df.weight.min()).round().astype('int')
bench_pd=[]
#find direct benchmarks which have highiest weights and create a phrase
#contained all benchmarks words with their weight frequency
for _, item in bench_df.iterrows():
    for j in range(item.weight_int):
        bench_pd.append(item.benchmark)
 #               bench_pd_shared.append(item.benchmark)
# normalize the benchmark words vector
A_bench=normalise(get_vector(word_dict,bench))
bench_pd_direct=['direct','finance'] #dirct benchmark list

direct_bench=normalise(get_vector(word_dict,bench_pd_direct))   
######################################








"""
-----main
"""

"""
global paramters setting 
""" 
    

#%%
#read files from internet pdf files
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
    
        

# a dataframe contins the processed input docuemtns
scor_df=pd.DataFrame(columns=['file_name'
                            ,'#_of_words',
                          'processing_time(s)',
                           'address','date', 'total_sim'])
aa=pd.read_csv(crawler_file)
start=time.time()
bench_catag=[['direct'],['finance'],['energy', 'electricity', 
       'heat',  'water', 'bio','waste'],['sustainable','carbon', 'env' ],
    [ 'vertical']]
for item in bench_catag:
    scor_df["%s"%(item)]=None
col_sc=["%s"%(item) for item in bench_catag]
for fname, rawText, txt, page_start, page_start_ln,page_bbox_list,page_bbox_list_whole, doc in \
    extract_text_from_folders_of_pdfs_zip([],crawler_file, stopwords1 ,max_files=2):#max_num_files):
    
    sentences,NER_df=find_sentences(rawText,NER=NER) 
    address=""
    year=0
    if(NER):
        address,year=evaluate_NER(NER_df)
#    evaluate_NER(NER_df)
    proj=pd.Series({'file_name':fname, 'org_doc':rawText, 'doc':txt, 'page_start_list':page_start,
             'start_ln':page_start_ln,'bbox_list':page_bbox_list,'bbox_list_whole':page_bbox_list_whole,'fitz_doc':doc})
    #calculate number of sentences siimlar to direct, benchmark and total
    scoring=words_score2(bench_w,A_bench,word_dict,bench_type_list,proj,tershold=.8,sub=bench_catag)
#    scoring = scoring.mean(axis=0)
    new_row=[fname,len(proj.doc.split()),time.time()-start
             ,address,year]+list(scoring)
    scor_df.loc[len(scor_df)]=new_row
    start=time.time()
#normalize factors by minmax scaler. 
# scaler = MinMaxScaler()
# scor=scaler.fit_transform(scor_df.iloc[:,3:6])
#this scor is calculated based on the intersection of all benchmarks and doc
# and divided by the total sim to benchmark
scor_df['intersect_score']=scor_df[col_sc].prod(axis=1)**(1/len(col_sc))#/scor_df['total_sim'].values
# tot_scor= (scor[:,0]**2+scor[:,1]**2+scor[:,2]**2)**.5/(3**.5)
# scor_df['total_score']=tot_scor
mina_df=pd.read_csv('mina_evaluation1.csv').set_index('file_name')
scor_df.set_index('file_name',inplace=True)
#mixed report with  reviewed descriptions

bb=pd.concat([mina_df,scor_df],axis=1,join='inner')
#%%

import plotly.express as px
#color_continues_scle is presented in:
#https://plotly.com/python/builtin-colorscales/
scor_df['refrence']=["<a href=\"%s\">%s</a>"%(aa,aa) for aa in scor_df.index]
fig = px.scatter(scor_df, x='total_sim', y='intersect_score',
	         size='#_of_words', color='total_sim',hover_name='refrence',
             hover_data={"['direct']":False,'#_of_words':False,'intersect_score':False}, log_x=False, size_max=60,
             color_continuous_scale='mygbm')
fig.write_html("bubble_chart.html")





#%%

list_of_hit=list(
    extract_text_from_folders_of_pdfs_zip([],crawler_file, stopwords1 ,max_files=100))
words_dis=list(np.zeros(len(list_of_hit)))#pd.DataFrame(columns=['word', 'distance'])
voc_w=[]
i=0
for fname, rawText, txt, page_start, page_start_ln,page_bbox_list,page_bbox_list_whole, doc in list_of_hit:
    words=[item for item in txt.split() if item in word_dict.keys()]
    words_vec=normalise(np.array([word_dict[item] 
                    for item in txt.split() if item in word_dict.keys()] )  )
    dist_matrix = (words_vec.dot(words_vec.T))
    dist_matrix[dist_matrix<.3]=0
    
    cum_dist = np.sum(dist_matrix, axis=0)/dist_matrix.shape[0]
    tmp_df=pd.DataFrame({'word': words, 'distance': cum_dist}).drop_duplicates(subset=['word'])
    words_dis[i]=tmp_df
    i+=1
    voc_w=voc_w+list(tmp_df['word'])
    voc_w=list(set(voc_w))
tresh=.8
q=[[] for item in range(len(voc_w))]
for j in range(len(voc_w)):
    item=   voc_w[j] 
    for i in range(len(words_dis)):    
        if(item in words_dis[i]['word'].values):
            q[j]=q[j]+list(words_dis[i][words_dis[i]['word']==item]['distance'].values)
l_q=[len(q[i]) for i in range(len(q))]
mean_dis=  [np.mean(q[i]) for i in range(len(q))]  
dd=pd.DataFrame({'word':voc_w,'freq':l_q,'values':q,'ave':mean_dis})        

"""
#proj_df['scoring']=0
scor=np.zeros((len(proj_df),4))
i=0
for proj_ind,proj in proj_df.iterrows():
    scoring=sents_score2(bench_w,A_bench,word_dict,bench_type_list,proj,tershold=.85,sub=[['direct']]).mean(axis=0)
    scor[i,1:]=scoring
    scor[i,0]=len(proj['sentence_start'])-1
    i+=1
scor_df=pd.DataFrame(data={'file_name':proj_df.index
                           ,'#_of_snetences':scor[:,0],
                           'total_sim':scor[:,1],'direct_bench':scor[:,2],
                           'reminded_bench':scor[:,3]})

#%%

all_project=pd.DataFrame({'project':proj_df.index,'Score':total_score_list})

f = []

dff=pd.DataFrame()
for (dirpath, dirnames, filenames) in os.walk("clustering"):
    for files in filenames:
        if(files.startswith(model_select+"+clustered+")):
#           print(files)
           df=pd.read_csv("clustering/"+files)
           dff[files]=df[df['average']!=-1]['average']
           
    f.extend(filenames)
    break
for col in dff.columns:
    mes=len(dff[dff[col]>.5][col])/len(dff[col].dropna())
    print(col, mes)
 #   print(col,dff.loc[:,col].sort_values(ascending=False)[:int(len(dff.loc[:,col])/10)].mean()
#    )
#%%
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(proj_df.loc['02-CUYAHOGA COUNTY UTILITY _ MICROGRIDS.pdf']['doc'])
wordcloud.to_file('CUYAHOGA.png') 
# plot the WordCloud image                      
import seaborn as sns; sns.set_theme()

ax = sns.heatmap(xdot)
"""