# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 09:16:16 2022

@author: Owner
"""
# import seaborn as sns
from sklearn_extra.cluster import KMedoids
# from io import BytesIO
from io import BytesIO
import requests
import fitz
import pathlib
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
#importing pymupdf library
#python -m pip install --upgrade pymupdf
import fitz
# import pathlib
import os
import multiprocessing as mp
# from datetime import datetime
# from gensim.models import Word2Vec
import collections
import re
from multiprocessing import get_context
import csv
from pyemd import emd

#https://analyticsindiamag.com/how-to-use-stanza-by-stanford-nlp-group-with-python-code/
#Note that for now installing Stanza via Anaconda does not work for Python 3.8.
# For Python 3.8 please use pip installation.
#pip install stanza
#stanza work better than spacy sentences tokenizer
#import stanza


import spacy
# load the pretrained Spacy model
#python -m spacy download en_core_web_lg --user
nlp_spacy = spacy.load("en_core_web_lg")
# for downloading the model following must be done
# conda install -c conda-forge spacy-model-en_core_web_sm
# conda install -c conda-forge spacy-model-en_core_web_lg
from Carlo_ngrams_tool.chunking_bforce_plus_space_add import\
    ngramming_bforce2,ngramming_bforce_bench

import stanza
# stanza.download('en')
# if NER:
nlp_stanza = stanza.Pipeline(lang='en',processors='tokenize,ner') 
# else:    
#nlp_stanza = stanza.Pipeline(lang='en',processors='tokenize') 

#normalize a vector
def normalise(A):
    lengths = (A**2).sum(axis=1, keepdims=True)**.5
    return A/lengths
#return word vectros from a vocabolary
def get_vector(word_dict,sub):
    return(np.array([word_dict[item] for item in sub if item in word_dict.keys()]))

def optimal_number_of_clusters(wcss):
    """
    The function take a list of wcss from kmean various # of cluster from 2 to max value 
    and find the point on the curve with the maximum distance to the line between
    first and last point of the curve as best # of cluster

    Parameters
    ----------
    wcss : List of the wcss value for the various kmean # of clusters  

    Returns
    -------
    int
        The optimal # of cluster .

    """
#    coordination of the line between the first and last wcss points
    x1, y1 = 2, wcss[0]
    x2, y2 = len(wcss), wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = ((y2 - y1)**2 + (x2 - x1)**2)**.5
        distances.append(numerator/denominator)

    return distances.index(max(distances)) + 2


"""
https://reposhub.com/python/natural-language-processing/boudinfl-pke.html
conda install -c conda-forge spacy-model-en_core_web_sm
"""
def find_dominanat_font(page):
    """
    Find the most pouplar (fint,size) in entire document 
    this could be best indicator of the most part of document and 
    the footer and noter usually have diffrence font,size attribute

    Parameters
    ----------
    doc : pyMUpdf documnet format
        
    examine an input pymupdf file and find the most applcable font in file
    it collect all bloacks applied fonts and for all of them find frequency of use. the font and size are 
    used together to find the most applied one.

    Returns
    -------
    the most applied tupple ('font_name',size).

    """
    pg_font=[]
    #itterate ove the document pages
    # for k in range( len(list(doc))):
    #     page =list(doc)[k]
        # access to the dict of the page
        #https://pymupdf.readthedocs.io/en/latest/textpage.html#TextPage.extractDICT
    dic =page.get_text("dict", sort=True)
    blks_font=[]
    blks_size=[]
    blks_no=[]
    #itterate over the block of the page
    for blks in dic['blocks']:
        blk_n=blks['number']
        #examine if the blks is a text only block
        if(blks['type']==0):
            lns_font=[]
            lns_size=[]     
            #itterate over lines of the block
            for lns in blks['lines']:  
                #iterrate over spans of each line
                for spns in lns['spans']:
                    #collect font type and size of each span
                    lns_size.append(int(spns['size']))
                    lns_font.append((spns['font'],int(spns['size'])))
            blks_font.append(lns_font)
            blks_size.append(lns_size)
            blks_no.append(blk_n)
    #find frequency of font,size applied in each page
    frequency = collections.Counter([x for y in blks_font for x in y]) 
#        print(frequency.most_common(1))
    pg_font.append(frequency.most_common(1))
    #find the most used font,size and return it as result
    dominanat_font = collections.Counter([x[0] for y in pg_font for x in y]).most_common(1)[0][0]
    return(dominanat_font)


def page_text(page):
    """
    find the blocks of text in page with domminant_font and concatinate them 
    to return the page content

    Parameters
    ----------
    page : page of document in pymupdf format
        DESCRIPTION.
    dominanat_font : dominate font in document as ('font_name',size)
        DESCRIPTION.

    Returns
    -------
    blk: the raw text in documnet
    start_ln:  starting pointer list of all lines in the returned text
    bbox_list:  list of quad coordination of bbox of span of each line in document .
    bbox_list_whole: list of the quad coordination of bbox of each line in document .
    
    """
    #get dict for the document it contins many information about the page 
    #content
    #https://pymupdf.readthedocs.io/en/latest/textpage.html#TextPage.extractDICT
    #https://pymupdf.readthedocs.io/en/latest/page.html?highlight=get_text#Page.get_text
    dic =page.get_text("dict", sort=True)
    blks_font=[]
    blks_no=[]
    blks_text=[]
    blks_bbox=[]
    blks_bbox_whole=[]
    #itterate over the blocks in document
    for blks in dic['blocks']:
        blk_n=blks['number']
        #if blks contains text and not graph
        if(blks['type']==0):
            lns_font=[]
            lns_text=[]
            lns_bbox=[]
            lns_whole_bbox=[]
            #itterate over lines in each block
            for lns in blks['lines']:  
                spn_font=[]
                spn_text=[]
                spn_bbox=[]                #itterate over span of each line and collect fon and size and text in spane
                for spns in lns['spans']:
                    spn_font.append((spns['font'],int(spns['size'])))
                    spn_text.append(spns['text'])
                    spn_bbox.append(spns['bbox'])
                #record the lines bbox with its text
                lns_bbox.append(spn_bbox)
                lns_font.append(spn_font)
                lns_text.append(spn_text)
                lns_whole_bbox.append(lns['bbox'])

                
            blks_font.append(lns_font) #block fonts
            blks_no.append(blk_n) #blcok sequence number
            blks_bbox.append(lns_bbox) #block bbox
            blks_bbox_whole.append(lns_whole_bbox)
            blks_text.append(lns_text) #block text
        else: #for graph blocks only insert null fields
            blks_bbox=blks_bbox+[""]
            blks_bbox_whole=blks_bbox_whole+[""]
            blks_font.append([(' ',0)])
            blks_no.append(blk_n)
            blks_text=blks_text+[""]           
    sel_blks_no=[]
    #itterate over collected blocks with its natual flow of block_no
    for kk in np.argsort(blks_no):
        # ind=blks_no[kk]
        # frequency=collections.Counter(blks_font[kk])
    #record only blocks that has the domminate font
#it assumend the main body only use mostly the dominant font                     
#        if(frequency.most_common(1)[0][0]==dominanat_font):
        sel_blks_no.append(kk)
 #        else:
 # #           print("ignore blk:%d"%(ind))
    sel_blks_no= np.array(sel_blks_no)

    blk=""
    start=0
    start_spn=[]
    bbox_list=[]
    bbox_list_whole=[]
    #ittterate over slected blcoks and record the starting postion in returned
    #string for each line and its bbox
    #conect lines with \n
    for k in sel_blks_no:
        for kk in range(len(blks_text[k])):
            
            for kkk in range(len(blks_text[k][kk])):
                start=len(blk)
                start_spn.append(start)
                bbox_list.append(blks_bbox[k][kk][kkk])
                blk=blk+" "+blks_text[k][kk][kkk]
            blk=blk+"\n"
            bbox_list_whole.append(blks_bbox_whole[k][kk])

    return(blk,start_spn,bbox_list,bbox_list_whole)

import geocoder
def evaluate_NER(NER_df):
    """
    

    Parameters
    ----------
    NER_df : DataFrame
        A dataframe contianse all available NER in the orginal document.

    Returns
    -------
    1. The most frequence geopgaphical address 
    2. The most recent year refrenced in document.

    """
    td=NER_df[NER_df['type']=='DATE']['text'].values
#    term=f'\b(19|20)\d{2}\b'
#collect dates which are contains prpoer date format and
#find the most repeatd one
    date_list=[]
    for item in td:
        try:
         date_list.append(pd.to_datetime(item))
        except:
            continue
    most_used_year=0
    most_recent_year=0
    if len(date_list)>0:
        year_dist=np.unique([item.year for item in date_list],return_counts=True)
        most_used_year=year_dist[0][year_dist[1].argmax()]
        most_recent_year=year_dist[0].max()
# collect georefrenced tokens and find the location detaisl by
#using geocoder if it represent some where    
    td=NER_df[NER_df['type']=='GPE']['text'].values
#    extract geo data and find the lcoation on map
#g.appress
#g.json['lng'],g.json['lat'],gg.city,gg.state,gg.county, g.country
# handling error https://geocoder.readthedocs.io/api.html#error-handling
    GPE_list=[]
    with requests.Session() as session:
        for item in td:
            try:
             g=geocoder.osm(item,session=session)
             if(g.ok):
                 GPE_list.append(g.address)         
            except:
                continue        
 #   add_list=[item['address'] for item in GPE_list]
    most_freq_add=""
    if(len(GPE_list)>0):
        most_freq_add=max(set(GPE_list),key=GPE_list.count)
    
    return most_freq_add,most_recent_year


def find_sentences(corpus_str,model='stanza',NER=False):
    """
    find sentenses boudaries by utilizing stanza or SPpacy library

    Parameters
    ----------
    corpus_str : string
        contins the entire input text.
    model      : string
        define which tokenize model will be used for finding sentences 'spacy' or 'stanza'
    Returns
    -------
    a list of starting position of each sentence in the input text
    it allso add a pointer to the len of text as the last element 
    of list.
    also if the STANZA applied a dataframe of doc segemnts and their
    NER type'
    """
    global nlp_stanza

    NER_df=pd.DataFrame(columns=['text','type'])
    #check which model needs to be apply for sentences extraction
    if(model=='spacy'):
        doc = nlp_spacy(corpus_str)
        nlp_spacy.max_length = 1500000
        # extract the sentences from the by looping over the processed file
        # it also extract the centroid vector of each sentence
        sentsp = []
#        sentsv = []
        sents_start = []
        for sentp in doc.sents:
            sentsp.append(str(sentp.text))  # list of sentences
#            sentsv.append(sentp.vector)    # list of the centroid sentence vector
    else:
        tx= nlp_stanza(corpus_str)
        sentsp=[item.text for item in tx.sentences] 
        #this part return NER type of each docuemnt segemnts
        #https://stanfordnlp.github.io/stanza/ner_models.html
        #it only collect the NER type and segment. however it could be matched
        #with the sentences to find the point of presentation in docuemnt 
        tmp=[[ent.text,ent.type] for sent in tx.sentences for ent in sent.ents]
        NER_df=pd.DataFrame(data=tmp,columns=['text','type'])
    # iterating over the senetences and find the startpoint of the sentences in the source document
    #
    start = 0
    sents_start = []
    for kk in range(len(sentsp)):
        index = corpus_str.find(sentsp[kk], start, len(corpus_str))
        start = index+len(sentsp[kk])
        sents_start.append(index)
    sents_start.append(len(corpus_str))
    return(sents_start,NER_df)


from zipfile import ZipFile
from urllib3.util.retry import Retry
import warnings
def read_file(source,max_files):
    """
    

    Parameters
    ----------
    source : String
        three options:
            1. a zip file contained all input file.
            2. a input csv file contaied crawler results as input
            3. a path contained all input file
    max_files : int
        maximum number of return files.

    Yield:
    -------
    pyMUpdf doc variable  and file refrence.

    """
    i=0
    if(source.endswith(".zip")):
        with ZipFile(source) as zf:
            for file in zf.namelist():
                if(i==max_files):
                    break
                if not file.endswith('.pdf'): # optional filtering by filetype
                    continue
                with zf.open(file) as f:
                    pdffile = f.read()
                    try:                        
                        doc = fitz.open(stream=pdffile, filetype="pdf")
                        i+=1
                        yield doc,file
                    except:
                        continue
    elif(source.endswith(".csv")):
        or_df=pd.read_csv(source)
        for ind,item in or_df.iterrows():            
            if(i==max_files):
                break 
            try:    
# the request adapted to try 3 times and wait .5s after each attempt                
# There is a certification problem with requests for SSL certification
# to solve problem I used a unsafe solution by verify=Flase
#however I got warning of unsecure get function 
#InsecureRequestWarning: Unverified HTTPS request
#to solve that I ignore warning as well. 
                session = requests.Session()
                retry = Retry(connect=3, backoff_factor=0.5)
                adapter = requests.adapters.HTTPAdapter(max_retries=retry)
                session.mount('http://', adapter)
                session.mount('https://', adapter)
                session.verify = False
                session.trust_env = False
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    request=session.get(item['web_link'], verify=False)
    #            request = requests.get(item['web_link'])
                get_date=request.headers['Date'] #extracted date
                content_len=int(request.headers['Content-Length'])
                Last_Modified =request.headers['Last-Modified']
                filestream = BytesIO(request.content)               
                doc = fitz.open(stream=filestream, filetype="pdf")
                i+=1
                yield doc,item['web_link']
            except:
                print("-------------------------")
                print("unable to downlaod file:",item['web_link'])
                continue
    else:
        for(root, dirs, files) in os.walk(source, topdown=True):
            if(i==max_files):
                break
            for fname in files:
                if(i==max_files):
                    break
                if (fname.endswith('.pdf')): 
                    fpath = os.path.join(root, fname)
                    try:                        
                        doc = fitz.open(fpath)  # open document
                        i+=1    
                        yield doc,fname
                    except:
                        continue



def extract_text_from_folders_of_pdfs_zip(bench_list,proj_dir, stopwords1,
                                      max_files=100,  doc_min_len=10):
    """
    extracting files from PDF_DTA_DIR and clean and apply bigram and trigram
    it also use a benchmark to keep those words in dictionary dispite
    rare use in input files. it returns clean text from each file in a dataframe 
    format. if the  input parapeter OrgDoc=True then the ouput dataframe will be contined
    orginal text file,cleaned one, a list of starting index of pages, starting sentence
    index list, and a pointer to PyMUpdf text page pointer


    Parameters
    ----------
    bench_list : list of the benchmarks
    proj_dir : directory contains pdf files
        DESCRIPTION.
    stopwords1 : stop words list
        DESCRIPTION.
    max_files : maximum files to read
        DESCRIPTION. The default is 100.
 
    doc_min_len : define a minumim len for input file 
                  default is 10
    OrgDoc : Bolean with default value False define if the return dataframe
            would be included more detailes about input text file
    Yield
    -------
    fname: input file name
    rawText : raw text of input file
    txt : cleaned text of inut file
    page_start: a list of start character index of each page in input text file
    page_start_ln: a list of start lines for each page
    page_bbox_list: a list of bbobx start index
    doc: a MuPDF instance cotians pdf pages fo accesing the pdf file 
    

    """
    i = 0

    for doc,fname in read_file(proj_dir,max_files):
        
                try:
#                    fpath = os.path.join(root, fname)
#                    print("....processing:",fname)
#                    doc = fitz.open(fpath)  # open document
                    rawText = ""  # open text output
                    page_start = []  # list of starting point of each page
#                    sentnce_start_list
                    # set to ignore warning and error messge from mupdf
                    fitz.TOOLS.mupdf_display_errors(False)
#                    dominanat_font =find_dominanat_font(doc)
                    page_start_ln=[]
                    page_bbox_list=[]
                    page_bbox_list_whole=[]
                    for page in doc:  # iterate the document pages
                        # get plain text (is in UTF-8)
                        text,start_ln,bbox_list,bbox_list_whole=page_text(page)
#                        text = page.get_text("text", sort=True)
                        if(text != ""):
                            # add the starting point of the new page to page list
                            page_start.append(len(rawText))
                            page_start_ln.append([x+len(rawText) for x in start_ln])
                            rawText = rawText+str(text)  # write text of page
                            page_bbox_list.append(bbox_list)                           
                            page_bbox_list_whole.append(bbox_list_whole)                           
                    # check if the lenght of text is at least doc_min_len charachter
                    if(len(rawText) > doc_min_len):
                        txt, _ = ngramming_bforce_bench(bench_list,rawText, stopwords1,
                                                   word_len=2, ngram_range=(2, 3),
                                                   bigram_base_frequency=.28,
                                                   trigram_base_frequency=.28)
                        print(i, '+<<', fname, ">> preprocessed with len=",len(txt))
                        i += 1                        
                        yield fname, rawText, txt, page_start, page_start_ln,page_bbox_list,page_bbox_list_whole, doc

                    else:
                        print(fname, "documnet is too short, not processed",
                              len(rawText))
                    print("===============================================")
                except Exception as e:
                    print(e, '-', fname, ': pdf is not readable')
                    pass
    print("\n\n########### End of input document files preprocessing#########\n\n")


def extract_text_from_folders_of_pdfs(proj_dir, stopwords1,
                                      max_files=100, bench=[], doc_min_len=10):
    """
    extracting files from PDF_DTA_DIR and clean and apply bigram and trigram
    it also use a benchmark to keep those words in dictionary dispite
    rare use in input files. it returns clean text from each file in a dataframe 
    format. if the  input parapeter OrgDoc=True then the ouput dataframe will be contined
    orginal text file,cleaned one, a list of starting index of pages, starting sentence
    index list, and a pointer to PyMUpdf text page pointer


    Parameters
    ----------
    PDF_DATA_DIR : directory contains pdf files
        DESCRIPTION.
    stopwords1 : stop words list
        DESCRIPTION.
    max_files : maximum files to read
        DESCRIPTION. The default is 100.
    bench : list of string
        the string list of benchmarks. The default is None.
    doc_min_len : define a minumim len for input file 
                  default is 10
    OrgDoc : Bolean with default value False define if the return dataframe
            would be included more detailes about input text file
    Returns
    -------
    rtn_dict : a dictionary of cleaned text which key is file name

    """
    i = 0

    for(root, dirs, files) in os.walk(proj_dir, topdown=True):
        if(i==max_files):
            break
        for fname in files:
            if(i==max_files):
                break
            if (fname.endswith('.pdf')): #only pdf will processed but it can process word and txt file as well
#                print("---processing<<%s>>"%(fname))
                try:
                    fpath = os.path.join(root, fname)
                    doc = fitz.open(fpath)  # open document
                    rawText = ""  # open text output
                    page_start = []  # list of starting point of each page
#                    sentnce_start_list
                    # set to ignore warning and error messge from mupdf
                    fitz.TOOLS.mupdf_display_errors(False)
#                    dominanat_font =find_dominanat_font(doc)
                    page_start_ln=[]
                    page_bbox_list=[]
                    page_bbox_whole_list=[]
                    for page in doc:  # iterate the document pages
                        # get plain text (is in UTF-8)
                        text,start_ln,bbox_list,bbox_list_whole=page_text(page)
#                        text = page.get_text("text", sort=True)
                        if(text != ""):
                            # add the starting point of the new page to page list
                            page_start.append(len(rawText))
                            page_start_ln.append([x+len(rawText) for x in start_ln])
                            rawText = rawText+str(text)  # write text of page
                            page_bbox_list.append(bbox_list)                           
                            page_bbox_whole_list.append(bbox_list_whole)                           
                    # check if the lenght of text is at least doc_min_len charachter
                    if(len(rawText) > doc_min_len):
                        txt, _ = ngramming_bforce2(rawText, stopwords1,
                                                   word_len=2, ngram_range=(2, 3),
                                                   bigram_base_frequency=.28,
                                                   trigram_base_frequency=.28)
                        print(i, '+<<', fname, ">> preprocessed")
                        i += 1                        
                        yield fname, rawText, txt, page_start, page_start_ln,page_bbox_list,page_bbox_whole_list, doc

                    else:
                        print(fname, "documnet is too short, not processed",
                              len(rawText))
                    print("===============================================")
                except Exception as e:
                    print(e, '-', fname, ': pdf is not readable')
                    pass
    print("\n\n########### End of input document files preprocessing#########\n\n")


def extract_text_from_folders_of_pdfs_complete(proj_dir, stopwords1,
                                      max_files=100, bench=[], doc_min_len=10, OrgDoc=False):
    if(OrgDoc):
        orgdoc_df = pd.DataFrame(
            columns=['file_name', 'org_doc', 'doc', 'page_start_list',
                     'sentence_start','start_ln','bbox_list','bbox_list_whole','fitz_doc'])
    else:
        orgdoc_df = pd.DataFrame(columns=['file_name', 'doc'])
        
    for fname, rawText, txt, page_start, page_start_ln,page_bbox_list,page_bbox_list_whole, doc in \
        extract_text_from_folders_of_pdfs_zip(bench,proj_dir, stopwords1,max_files, doc_min_len):
        if(OrgDoc):
            sentences,NER_df=find_sentences(rawText)
            orgdoc_df.loc[len(orgdoc_df)] = [
                fname, rawText, txt, page_start, sentences,
               page_start_ln,page_bbox_list,page_bbox_list_whole, doc]
        else:
            orgdoc_df.loc[len(orgdoc_df)] = [
                fname, txt.split()]
    return orgdoc_df.set_index('file_name')
        

def distribution_cal(cosin_matrix):
    """
#make a 2 bins histogram to implement the probability distribution of 
# the words similarity to each benchmarks
# as the cosine parameter has a variance between -1,1 the 2 bin probability
#distribution has 2 bin (-1,0) and (0,1) the similarity of two vector would be
# the bin (0,1) as second bin. this will be show the similarity
#this part extract as similarity of the doc to specific benchmark
#    

    Parameters
    ----------
    cosin_matrix : TYPE
        A matrix of cosine similarity between two comparing set of vectors
        it returns distribution of cosine more than 0 for each row.

    Returns
    -------
    it returns distribution of cosine more than 0 for each row as a numpy
    array.

    """
    bins=2
    hist= np.zeros((cosin_matrix.shape[0],bins))
    for k in range(hist.shape[0]):
        hist[k,:],rng=np.histogram(
        cosin_matrix[k,:],
        bins=bins,
        range=(-1, 1),
        density=True) 
    return(hist[:,1])    


# creat cluster for benchmarks
def bench_clustring(bench_df,word_dict,max_cluster=50):
    """
    accept the benchmarks dataframe and add a catagory as 'class' and also 
    medoids pharse for each bencmmarks catagory. 

    Parameters
    ----------
    bench_df : Dataframe from the benchmarks
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # normalize the benchmark words vector
    bnch=bench_df.benchmark
    A_bench = normalise(np.array([word_dict[item] for item in bnch if item in word_dict.keys()]))
    xdot = 1 - A_bench.dot(A_bench.T)
    wcss = []  # the list of inertia for each number of cluster
    for i in range(2, max_cluster):
        
        kmedoids = KMedoids(n_clusters=i, random_state=0).fit(xdot)
    #    print(kmedoids.inertia_, "for kmedoids with n_cluster=",i)
        wcss.append(kmedoids.inertia_)

    wcss = np.array(wcss)
    # find the optimal # of cluster
    n_cluster = optimal_number_of_clusters(wcss)
    print("---Clustering Benchmarks")
    # using kmedoids technic to find the clustring lables for sentenses
    kmedoids = KMedoids(n_clusters=n_cluster, random_state=0).fit(xdot)    
    
    bench_df['class'] = kmedoids.labels_

    # creat a datarame from cluster labels , number of sentences in each label, the medoids sentence
    # and the index of medoids sentence in the orginal document
    unique, counts = np.unique(kmedoids.labels_, return_counts=True)
    print("%d benchmarks are clustered into %d clusters\nNumber of bench in each cluster is:"%(len(bnch),len(unique)),counts)

    bench_df['class_mediods_bench'] = [bnch[kmedoids.medoid_indices_[x]] for x in bench_df['class'].values]



#%% 


#This WMD version use words frequency and is most standard version
def wmdistance3(model, document1, document2, norm=True,extra_mass_penalty=-1):
    """Compute the Word Mover's Distance between two documents.

    When using this code, please consider citing the following papers:

    * `Ofir Pele and Michael Werman "A linear time histogram metric for improved SIFT matching"
      <http://www.cs.huji.ac.il/~werman/Papers/ECCV2008.pdf>`_
    * `Ofir Pele and Michael Werman "Fast and robust earth mover's distances"
      <https://ieeexplore.ieee.org/document/5459199/>`_
    * `Matt Kusner et al. "From Word Embeddings To Document Distances"
      <http://proceedings.mlr.press/v37/kusnerb15.pdf>`_.

    Parameters
    ----------
    document1 : list of str
        Input document.
    document2 : list of str
        Input document.
    norm : boolean
        Normalize all word vectors to unit length before computing the distance?
        Defaults to True.
   
    extra_mass_penalty : float
   Compute WMD. https://github.com/wmayner/pyemd
   The penalty for extra mass in EMD function . If you want the resulting distance to be a metric, 
   it should be at least half the diameter of the space (maximum possible distance between any two points).
   If you want partial matching you can set it to zero (but then the resulting distance is not guaranteed to be a metric).
   The default value is -1.0, which means the maximum value in the distance matrix is used.
    
    Returns
    -------
    float
        Word Mover's distance between `document1` and `document2`.

    Warnings
    --------
    This method only works if `pyemd <https://pypi.org/project/pyemd/>`_ is installed.

    If one of the documents have no words that exist in the vocab, `float('inf')` (i.e. infinity)
    will be returned.

    Raises
    ------
    ImportError
        If `pyemd <https://pypi.org/project/pyemd/>`_  isn't installed.

    """
    # If pyemd C extension is available, import it.
    # If pyemd is attempted to be used, but isn't installed, ImportError will be raised in wmdistance
    from pyemd import emd

    # Remove out-of-vocabulary words.
    # len_pre_oov1 = len(document1)
    # len_pre_oov2 = len(document2)
    document1 = [token for token in document1 if token in model]
    document2 = [token for token in document2 if token in model]
    # diff1 = len_pre_oov1 - len(document1)
    # diff2 = len_pre_oov2 - len(document2)
    # if diff1 > 0 or diff2 > 0:
    #     print('Removed %d and %d OOV words from document 1 and 2 (respectively).'%( diff1, diff2))

    if not document1 or not document2:
        print("At least one of the documents had no words that were in the vocabulary.")
        return float('inf') #return a vast amount as distance if one docuemnt is empty
#calcuclate the frequency of words in two documents
    vectorizer = CountVectorizer()
    corpus=[' '.join(document1),' '.join(document2)]
    X = vectorizer.fit_transform(corpus).toarray().astype('float64')
    d1=X[0] 
    d2=X[1]
    if(norm):
        d1/=len(document1)
        d2/=len(document2)
    dictionary=list(vectorizer.get_feature_names_out())

 #   dictionary = document1+ document2
    vocab_len = len(dictionary)
    if vocab_len == 1:
        # Both documents are composed of a single unique token => zero distance.
        return float(0.0)
    vdoc1=normalise(get_vector(model,dictionary))
    #calcualate the distance matrix for vocabulary words
    
    cosin_matrix=1-vdoc1.dot(vdoc1.T)
    # I have used matrix operation instead of for loop that improve speed
    #more than 10 times
    m1 =np.array([dictionary for i in range(len(dictionary))])
    m2 =np.array( [[dictionary[i]]*len(dictionary) for i in range(len(dictionary))])
    cond1=~(np.isin(m1,document1) & np.isin(m2, document1) & ~np.isin(m1, document2) & ~np.isin(m2,document2))
    cond2=~(np.isin(m1,document2) & np.isin(m2, document2) & ~np.isin(m1, document1) & ~np.isin(m2,document1))
    distance_matrix=np.multiply(cosin_matrix , np.multiply(cond1.astype('int'),cond2.astype('int')))
   
    # if words pari are only in document 1 set theri distance to zero
    # if word pairs are only in document 2 set their distance to zero
    # for i in range(len( dictionary)):
    #     t1=dictionary[i]
    #     for j in range(len( dictionary)):
    #         t2=dictionary[j]
    #         if t1 in document1 and t2  in document1:
    #             if t1 not in document2 and t2 not in document2:
    #                 cosin_matrix[i,j]=0
    #         if t1 in document2 and t2  in document2:
    #             if t1 not in document1 and t2 not in document1:
    #                 cosin_matrix[i,j]=0 
    # distance_matrix = cosin_matrix.astype('float64')

    if abs(np.sum(distance_matrix)) < 1e-8:
        # `emd` gets stuck if the distance matrix contains only zeros.
#        print('The distance matrix is all zeros. Aborting (returning inf).')
        return float(0) #distance matrix is all zero!
    ee=emd(d1, d2, distance_matrix, extra_mass_penalty)
#    print("--- %s seconds ---" % (time.time() - start_time))
    
    return ee 


# import time
# from multiprocessing import Pool



def cluster_sentences(corpus_str,cleaned_str,sents_start,word_dict,max_cluster=50):
    """
    Cluster sentences of a corpus by applying kmedoids algorithms and average sentences
    words vectros as a sentence vector presentation

    Parameters
    ----------
    corpus_str : string
        the input uncleaned corpus.
    cleaned_str : TYPE
        the input cleaned corpus.
    sents_start : TYPE
        list of the starting indice of corpus sentences.
    word_dict   : dictionary
        a diconary of all vocabullary words and their vectors

    Returns
    -------
    clusterd_df_kmedoids: dataframe
                a dataframe contines uncleaned snetnces and their cluster label
    
    clusterd_df_stats: data frame
                a dataframe contines cluster labels and the mediods sentences of
                each cluster

    """    
# this section iterate over sentences in corpus and  find the word2vec presentation of existing
# word in sentences then caclualte average weighted vector by using content as weight of
# each word   . for sentences with no words in vocab , no vector is recorded.

    sentsp_pure = []  # list of sentences that have at least one word in vocab
    sents_list = []   # list of index of the sentences in orginal sentences list
    # list of the average weighted word2vec vector for the sentence
    sents_weighted_vector = []
    for ind in range(len(sents_start)-1):
        start = sents_start[ind]
        end = sents_start[ind+1]
        focus_sents = cleaned_str[start:end].split(' ')
    
        # select the words from sentence which have entery in vocab
        sub_sent = list(set(focus_sents).intersection(set(word_dict.keys())))
        # document words vectors  multiply by the content measure
        # which present the rareness and calculate the average of the weighted sentences vectors
#########  I decided to not use CONTENT factor!!!
        if(len(sub_sent) != 0):
            # w_vec = np.sum(w2vec_model.wv[sub_sent].T * voc_cnt.loc[sub_sent].content.values,
            #                axis=1) / voc_cnt.loc[sub_sent].content.sum()
            vect_sub_sent=np.array([word_dict[item] for item in sub_sent if item in word_dict.keys()])
            w_vec = np.average(vect_sub_sent,axis=0)
            sents_weighted_vector.append(w_vec)
            sents_list.append(ind)
            sentsp_pure.append(cleaned_str[start:end])
    
    # creat an numpy array from the weighted centroid vector of all sentences
    X = np.array(sents_weighted_vector)
    
    
    ################ cluster the sentences
    
    # this part applies Kmedoids clustering method to find the nearest pointd to each other
    # in a form of distance matrix. distnace matrix has been produced from the cosine similarity
    # measure. the xdot is cosine similarity matrix between all words in document
    # I belive as the k-means use the euclidain distance and use average distance
    # it is not good measure for comparing word2vec vectors. a better approch would be using
    # cosine simialrity matrix and apply  Kmedoids as technic
    # I have transfer the cosine similarity to distance by subtracting it form 1
    # 1-cosin would be a measure that the least one would be 0 and highiest one 2
    
    
    
    # def index_max(a):
    #     return(np.unravel_index(np.argmax(a, axis=None), a.shape))
    
#    print("--------estimate best number of cluster by applying elbow method ")
#normalize the sentence vectors
    nx = normalise(X)
#find the cosine similarity between the e=sentences by dot product and
#change the measure to a distance 
    xdot = 1 - nx.dot(nx.T)
# wcss is the list of the sum of squared distance between each point 
#and the centroid in a cluster for each number of cluster      
    wcss = []  
    for i in range(2, max_cluster):
        
        kmedoids = KMedoids(n_clusters=i, random_state=0).fit(xdot)
    #    print(kmedoids.inertia_, "for kmedoids with n_cluster=",i)
        wcss.append(kmedoids.inertia_)
    
    wcss = np.array(wcss)
    # find the optimal # of cluster
    n_cluster = optimal_number_of_clusters(wcss)
#    print("Best number of cluste would be:",n_cluster)
    # using kmedoids technic to find the clustring lables for sentenses
    kmedoids = KMedoids(n_clusters=n_cluster, random_state=0).fit(xdot)
    # creat a dataframe from setences and their cluster label
    clusterd_df_kmedoids = pd.DataFrame({'sentence': sentsp_pure})
    clusterd_df_kmedoids['class'] = kmedoids.labels_
    all_sents_clusters=np.array([-1]*(len(sents_start)-1))
    all_sents_clusters[sents_list]=kmedoids.labels_
    # create a dataframe from cluster labels , number of sentences in each label, the medoids sentence
    # and the index of medoids sentence in the orginal document
    unique, counts = np.unique(kmedoids.labels_, return_counts=True)
    
    clusterd_df_stats = pd.DataFrame({'label': unique})
    clusterd_df_stats['count'] = counts
    clusterd_df_stats['medoids_sentence'] = np.array(
        sentsp_pure)[kmedoids.medoid_indices_]
    clusterd_df_stats['mediods_sentence_indice'] = kmedoids.medoid_indices_
    return(all_sents_clusters,clusterd_df_kmedoids,clusterd_df_stats)

def cluster_sentences_wmd(corpus_str,cleaned_str,sents_start,word_dict,max_cluster=50,Multi=False):
    """
    

    Parameters
    ----------
    corpus_str : string
        the input uncleaned corpus.
    cleaned_str : TYPE
        the input cleaned corpus.
    sents_start : TYPE
        list of the starting indice of corpus sentences.
    word_dict   : dictionary
        a diconary of all vocabullary words and their vectors
    max_cluster: int
    maximum number of cluster defalut is 50
    
    Returns
    -------
    clusterd_df_kmedoids: dataframe
                a dataframe contines uncleaned snetnces and their cluster label
    
    clusterd_df_stats: data frame
                a dataframe contines cluster labels and the mediods sentences of
                each cluster

    """    
# this section iterate over sentences in corpus and  find the word2vec presentation of existing
# word in sentences then caclualte average weighted vector by using content as weight of
# each word   . for sentences with no words in vocab , no vector is recorded.
#    start_time=time.time()
    sentsp_pure = []  # list of sentences that have at least one word in vocab
    sents_list = []   # list of index of the sentences in orginal sentences list
#    sents_weighted_vector = []
    #put space instead of words that not in dictionary in a corpus
    extract_words =list(set(cleaned_str.split())-set(word_dict.keys()))
    for item in extract_words:
        if len(item)>0:
            cleaned_str=re.sub(r'\b%s\b'%(item), ' '*len(item), cleaned_str)
    #list the snetences with at least one word length
    for ind in range(len(sents_start)-1):
        start = sents_start[ind]
        end = sents_start[ind+1]
        focus_sents = cleaned_str[start:end].split()
    
        if(len(focus_sents) != 0):
            sents_list.append(ind)
            sentsp_pure.append(cleaned_str[start:end])

#    print(time.time()-start_time)
# create an numpy array distance matrix from the wmdistance  
# between all sentences

    wmd_matrix=np.zeros(shape=(len(sents_list),len(sents_list)))
    if(Multi):
        arg_list=[]
        for i in range(len(sents_list)):
            si = cleaned_str[sents_start[sents_list[i]]:sents_start[sents_list[i]+1]].split()                        
            for j in range(len(sents_list)):
                if(i>j):
                    sj = cleaned_str[sents_start[sents_list[j]]:sents_start[sents_list[j]+1]].split()                            
                    arg_list.append((word_dict, si, sj,True,-1))
#        with Pool() as pool:
        with get_context("spawn").Pool() as pool:
        # prepare arguments
        
        # call the same function with different data in parallel
            results=[result for result in pool.starmap(wmdistance3, arg_list)]
            # report the value to show progress
        ind=0            
        for i in range(len(sents_list)):
            for j in range(len(sents_list)):
                if(i>j):
                    wmd_matrix[i,j]=results[ind]
                    wmd_matrix[j,i]=results[ind]
                    ind+=1
    else:    
        for i in range(len(sents_list)):
            si = cleaned_str[sents_start[sents_list[i]]:sents_start[sents_list[i]+1]].split()                        
            for j in range(len(sents_list)):
                if(i>j):
                    sj = cleaned_str[sents_start[sents_list[j]]:sents_start[sents_list[j]+1]].split()                            
                    distance =  wmdistance3(word_dict, si, sj, norm=True,extra_mass_penalty=-1)
                    wmd_matrix[i,j]=distance
                    wmd_matrix[j,i]=distance
    #                print("sentences distance (%d,%d) from %d                      \r"%(i,j,
    #                       len(sents_list)),end='')
    
#    print(time.time()-start_time)

    xdot = wmd_matrix
    
    
    ################ cluster the sentences
    
    # this part applies Kmedoids clustering method to find the nearest pointd to each other
    # in a form of distance matrix. distnace matrix has been produced from the cosine similarity
    # measure. the xdot is cosine similarity matrix between all words in document
    # I belive as the k-means use the euclidain distance and use average distance
    # it is not good measure for comparing word2vec vectors. a better approch would be using
    # cosine simialrity matrix and apply  Kmedoids as technic
    # I have transfer the cosine similarity to distance by subtracting it form 1
    # 1-cosin would be a measure that the least one would be 0 and highiest one 2
    
    
    
    # def index_max(a):
    #     return(np.unravel_index(np.argmax(a, axis=None), a.shape))
    
#    print("--------estimate best number of cluster by applying elbow method ",)

# the list of the sum of squared distance between each point 
#and the centroid in a cluster for each number of cluster    
    wcss = []  
    for i in range(2, max_cluster):  
        kmedoids = KMedoids(n_clusters=i, random_state=0).fit(xdot)
    #   print(kmedoids.inertia_, "for kmedoids with n_cluster=",i)
        wcss.append(kmedoids.inertia_)
#    print(time.time()-start_time)
    
    wcss = np.array(wcss)
    # find the optimal # of cluster
    n_cluster = optimal_number_of_clusters(wcss)
#    print("Best number of cluster would be:",n_cluster)
    # using kmedoids technic to find the clustring lables for sentenses
    kmedoids = KMedoids(n_clusters=n_cluster, random_state=0).fit(xdot)
    # creat a dataframe from setences and their cluster label
    clusterd_df_kmedoids = pd.DataFrame({'sentence': sentsp_pure})
    clusterd_df_kmedoids['class'] = kmedoids.labels_
    all_sents_clusters=np.array([-1]*(len(sents_start)-1))
    all_sents_clusters[sents_list]=kmedoids.labels_
    # creat a datarame from cluster labels , number of sentences in each label, the medoids sentence
    # and the index of medoids sentence in the orginal document
    unique, counts = np.unique(kmedoids.labels_, return_counts=True)
    
    clusterd_df_stats = pd.DataFrame({'label': unique})
    clusterd_df_stats['count'] = counts
    clusterd_df_stats['medoids_sentence'] = np.array(
        sentsp_pure)[kmedoids.medoid_indices_]
    clusterd_df_stats['mediods_sentence_indice'] = kmedoids.medoid_indices_
    return(xdot,all_sents_clusters,clusterd_df_kmedoids,clusterd_df_stats)




def sentence_distance(word_dict,proj,bench_pd,direct_bench,Multi=False):
    # select the words from sentence which have entery in vocab
    corpus_str = proj['org_doc']
    cleaned_str = proj['doc']
    sents_start = proj['sentence_start']

        
    sentsp_pure = []  # list of sentences that have at least one word in vocab
#        sents_list = []   # list of index of the sentences in orginal sentences list
#        sents_weighted_vector = []  # list of the  weighted maximum similarity to benchmark
    sents_benchw = [] # list of sentences average similarity to benchmark 
    sents_direct=[] # maximum simlarity between sentence's words and direct benchmarks
    sent_page=[] # sentences page numbe list
    for ind in range(len(sents_start)-1):
#        print("--- %s seconds ---for %d" % (time.time() - start_time,ind))

        # loop over the sentenses
        start = sents_start[ind]
        end = sents_start[ind+1]
        # page_ind_start=np.sum(page_start<start)-1
        # page_ind_end=np.sum(page_start<end)-1
        focus_sents = cleaned_str[start:end].split()
        sub_sent = list(set(focus_sents).intersection(set(word_dict.keys())))
    # sentence words vectors  multiply by the content measure
    # which present the rarreness and calculate the average of the weighted sentences vectors
        if(len(sub_sent) != 0):
            # WMD distnace between benchmark words list and eachsentence
            #bench_pd is a word list of benchmark which weight of each benchmarks
            #is reflected by repeating the word
            sentence_dist= wmdistance3(word_dict, focus_sents, bench_pd,
                          norm=True,extra_mass_penalty=-1)   
            
            sent_words = normalise(get_vector(word_dict, sub_sent))
            cosin_max=direct_bench.dot(sent_words.T).max() # maximum cosine matrix between dircet benchmarks and sents words  
            
        else : # null sentences are taged as -1 
            sentence_dist=-1
            cosin_max=-2
        sents_benchw.append(sentence_dist) 
        sents_direct.append(cosin_max)
        # original sentence in document
        sentsp_pure.append(corpus_str[start:end])
        # add page number for each sentence
        sent_page.append(np.sum(proj.sentence_start[ind]>=np.array(proj.page_start_list)))
    return(sents_benchw,sents_direct,sentsp_pure,sent_page) 





def cluster_similarity(word_dict, clusterd_sents, bench_pd,Multi=False):
    sim_cluster=np.zeros(shape=(len(np.unique(clusterd_sents['class']))-1)) 
    sim_cluster.fill(-1)
    sub_cluster_list=[]
    for ind in range(len(np.unique(clusterd_sents['class']))-1):
        cluster_corpus=" ".join(clusterd_sents[clusterd_sents['class']==ind]['sentence']).split()
        sub_cluster=[item for item in cluster_corpus if item in word_dict.keys()]
        # select the words from sentence which have entery in vocab
#        sub_cluster = list(set(cluster_corpus).intersection(set(word_dict.keys())))        
        sub_cluster_list.append(sub_cluster)
    for ind in range(len(sub_cluster_list)):
        sub_cluster=sub_cluster_list[ind]    
        if(len(sub_cluster) != 0):
                cluster_dist= wmdistance3(word_dict, sub_cluster, bench_pd,
                                          norm=True,extra_mass_penalty=-1)
                sim_cluster[ind]=  cluster_dist
    return(sim_cluster)



def hlight(page,search_term,start):
    """
    Parameters
    ----------
    page : TYPE
        pyMuPDF page object  .
    search_term : string
        an string which want to highlited in page.
    start : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """    
    if(len(search_term)>1):
        matching_val_area = page.search_for(search_term,quads=True)
        if(len(matching_val_area)!=0):
        #                    print(matching_val_area)
            highlight = page.add_highlight_annot(matching_val_area)
            highlight.update()  
    
        else:
            for sub_search_term in search_term.split('\n'):
    
                if(len(sub_search_term)>5): # at least two charachter would be there
                    matching_val_area = page.search_for(sub_search_term,quads=True)
                    if(len(matching_val_area)!=0):
                        highlight = page.add_highlight_annot(matching_val_area)
                        highlight.update() 
                    else: #one more time try on finding the subterm by ignoring last character
                        matching_val_area = page.search_for(sub_search_term[:-1],quads=True)
                        if(len(matching_val_area)!=0):
                            highlight = page.add_highlight_annot(matching_val_area)
                            highlight.update() 
                        else:    
                           print("NO SUBMATCH FOUND:<<",sub_search_term,">>",start)



def highlight_sents(proj_dir,doc_ind,proj_df,all_sents,model_select='word2vec'):
    """
    in ans specifc input file in proj_df with doc_ind all_sents selected
    are highlited in pyMuPDF object and the result wrote to a outputfile 
    in a director "Highlighted_files/HighLighted++%s" which %s is the same input
    file name

    Parameters
    ----------
    proj_dir : DataFrame
        input project directory name  .
    doc_ind : TYPE
        doc index in proj_df.
    proj_df : TYPE
        Dataframe contains input document files.
    all_sents : TYPE
        sentence which must be highlited.

    Returns
    -------
    None.

    """
 #   fpath = os.path.join(proj_dir, "++%s"%(doc_ind))
    
#    pdfdoc = fitz.open(fpath)
    output_buffer = BytesIO()
    path = pathlib.Path('Highlighted_files')
    path.mkdir(parents=True, exist_ok=True)
    output_file="Highlighted_files/%s_HighLighted++%s"%(model_select,doc_ind)
#    print("create highlited outpout file:",output_file)

    sents_start = proj_df['sentence_start'][doc_ind]
    page_start=np.array(proj_df['page_start_list'][doc_ind])
    pdfdoc=proj_df['fitz_doc'][doc_ind]
    for ind_sent in range(len(all_sents)):
        if(all_sents[ind_sent]):
            start = sents_start[ind_sent]
            end   = sents_start[ind_sent+1]
#            print(start,end,ind_sent)
            strt_ln=[x for y in proj_df.loc[doc_ind,'start_ln'] for x in y]
            bbox_l= [x for y in proj_df.loc[doc_ind,'bbox_list'] for x in y]
#            bbox_l= [x  for x in proj_df.loc[doc_ind,'bbox_list_whole'] ]
            for k1 in range(len(strt_ln)-1):  

                overlap_s=strt_ln[k1]
                overlap_e=strt_ln[k1+1]
                if(start>overlap_s):overlap_s=start
                if(end<overlap_e): overlap_e=end
                s=""
                if(overlap_s<overlap_e):
                    s=proj_df.loc[doc_ind,'org_doc'][overlap_s:overlap_e]
                if (len(s.split())>0) & (start<strt_ln[k1+1]) & (end>strt_ln[k1]):
                    # if((strt_ln[k1]>=start) & (strt_ln[k1+1]<=end)) |\
                    #   ((strt_ln[k1]<=start) & (strt_ln[k1+1]>=start) )|\
                    #   ((strt_ln[k1]<end) & (strt_ln[k1+1]>end)):   
                    # if(start<strt_ln[k1+1] & end>strt_ln[k1]):
                        page_ind=np.sum(page_start<=strt_ln[k1])-1
                        page  = pdfdoc[int(page_ind)]
                        highlight = page.add_highlight_annot(bbox_l[k1])
                        if(highlight!=highlight):
                            print("no highlight found!",page_ind,
                                  proj_df['org_doc'][doc_ind][strt_ln[k1]:strt_ln[k1+1]])

    pdfdoc.save(output_buffer)
#   pdfdoc.close()       

    with open(output_file, mode='wb') as f:
        f.write(output_buffer.getbuffer() )             


def vector_ave(ppp,tershold,bench_w):
    pmax=ppp.max(axis=1)[ppp.max(axis=1)>tershold]
    pmax_w=ppp.argmax(axis=1)[ppp.max(axis=1)>tershold]
    
    tot= np.sum(bench_w[pmax_w]*pmax)
    if tot==0 :
        average=0
    else:
        average=tot/ np.sum(bench_w[pmax_w])
    return(tot,average)
def vector_ave1(ppp,tershold,bench_w,bench_type_list,excloude):
    if(len(excloude)==0):
        
        pmax=ppp.max(axis=1)[ppp.max(axis=1)>tershold]
        pmax_w=ppp.argmax(axis=1)[ppp.max(axis=1)>tershold]
        tot= bench_w[pmax_w]*pmax

    else:
        term=np.isin(np.array(bench_type_list),excloude)
        ppp1=ppp[:,term]
        pmax=ppp1.max(axis=1)[ppp1.max(axis=1)>tershold]
        pmax_w=ppp1.argmax(axis=1)[ppp1.max(axis=1)>tershold]
        tot= bench_w[term][pmax_w]*pmax
    if(len(tot)==0):
            return(0)
    else:
        return(tot.max())
        

        
def words_score2(bench_w,A_bench,word_dict,bench_type_list,proj,tershold=.5,sub=[['direct','finance']]):       
    txt=proj.doc.split()  
    words_benchw= np.zeros((len(sub)+1))     
    sub_txt  =[token for token in txt if token in word_dict] 
    if(len(sub_txt) != 0):    
        B = normalise(get_vector(word_dict,sub_txt))
        ppp = B.dot(A_bench.T)
        ppp[ppp<tershold]=0
        excloude=[]
        bench_ave=np.sum(ppp.mean(axis=0)*bench_w)/np.sum(bench_w)*len(bench_w)
        words_benchw[0]=bench_ave
        if(len(sub)!=0):
            for k in range(len(sub)):
                excloude=sub[k]
                term=np.isin(np.array(bench_type_list),excloude)
                bench_ave=np.sum((ppp.mean(axis=0)*bench_w)[term])/np.sum(bench_w[term])*len(bench_w[term])
                words_benchw[k+1]=bench_ave        
    return(words_benchw)
def sents_score2(bench_w,A_bench,word_dict,bench_type_list,proj,tershold=.5,sub=[['direct','finance']]):
    
    # list of sentences average similarity to benchmark 
    sents_benchw=np.zeros((len(proj.sentence_start)-1,len(sub)+1))
    for ind in range(len(proj.sentence_start)-1):  # loop over the sentenses
        start = proj.sentence_start[ind]
        end = proj.sentence_start[ind+1]
    
        # split words on cleaned version of sentence
        focus_sents = proj.doc[start:end].split() 
        sub_sent  =[token for token in focus_sents if token in word_dict] 
#        avg=0
        if(len(sub_sent) != 0):
            # normalized sentence vectors
            B = normalise(get_vector(word_dict,sub_sent))
            ppp = B.dot(A_bench.T)
            excloude=[]
            sents_benchw[ind][0]=vector_ave1(ppp, tershold,bench_w,bench_type_list,excloude)
            if(len(sub)!=0):
                for k in range(len(sub)):
                    excloude=sub[k]
                    average1=vector_ave1(ppp, tershold,bench_w,bench_type_list,excloude)
                    sents_benchw[ind][k+1]=average1
            #     excloude=[]
            #     [excloude.extend(sub[k]) for k in range(len(sub))]
            #     excloude=list(set(bench_type_list)-set(excloude))
            #     sents_benchw[ind][len(sub)+1]=vector_ave1(ppp, tershold,bench_w,bench_type_list,excloude)
            else:
                excloude=[]
                sents_benchw[ind][1]=vector_ave1(ppp, tershold,bench_w,bench_type_list,excloude)
    #            sents_benchw[ind][0]=avg         #use average cosine

    return(sents_benchw)            
            
def glove2dict(glove_filename):
    """
    read a glove pretrained model dictionary according to stanford defined format
    https://nlp.stanford.edu/projects/glove/

    Parameters
    ----------
    glove_filename : string
        file name of input glove model.

    Returns
    -------
    embed : dictionary 
        return a dictionary of words and their related vector.

    """
    with open(glove_filename, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed

def most_similar(term,dictionary,n,dic_is_normal=False):
    """
   search for a specific term and try to find n most 
   similar terms in dictionary by applying cosine similarity and 
   normal vector concept

    Parameters
    ----------
    term : string
        a word looking for similar term to it.
    dictionary : TYPE
        DESCRIPTION.
    n : int
        number of similar word return.
    dic_is_normal: Boolean  
        wether dictionary is already normalized or not default False
    Returns
    -------
    a list of n most similar words and their similarity degree in cosine .

    """
    try:
        ind=list(dictionary.keys()).index(term)
        vect=np.array(list(dictionary.values()))
        if(dic_is_normal==False):
            vect =normalise(vect)
        cosine=vect.dot(vect[ind])
        cosine_ind=np.argsort(cosine)
        return(list(zip(np.array(list(dictionary.keys()))[cosine_ind[-n:]],
                  cosine[cosine_ind[-n:]])))
    except:
        return("")
def creat_bench(file_bench,Training=False,vocab={}):
    """
    return a list of benchmarsk in file_bench

    Parameters
    ----------
    file_bench : string
        input file name.
    Training : Bolean
        Set to True if iti is training mode
    vocab :  dictionary  
        vocaboulary dictionary of words

    Returns
    -------
    st: benchmark numpy list of string.
    wt : weight of each benchmark
    typ_list: type of each benchmark

    """
    df = pd.read_csv(file_bench)
    st = []
    wt = []
    typ_list=[]
    columns_words = [0]+list(range(3, len(df.columns)))
    for k in range(len(df)):
        
        for j in columns_words:
            if(df.iloc[k, j] == df.iloc[k, j]):
                js = '_'.join(df.iloc[k, j].strip().lower().split())
                if(js not in st):
                    st.append(js)
                    wt.append(df['weight'][k])
                    typ_list.append(df['type'][k])
    
    if(not Training):
        bench_found = list(set(st).intersection(set(vocab)))
        
        # ignore benchmarks which is not in vocabulary
        wt = [wt[i] for i in range(len(st)) if st[i] in bench_found]
        typ_list =[typ_list[i] for i in range(len(st)) if st[i] in bench_found]
        st = [st[i] for i in range(len(st)) if st[i] in bench_found]
        # normalized bench marks weight between minimum and 1.
        wt /= np.max(wt)
    return(st, wt,typ_list)
################################################################

