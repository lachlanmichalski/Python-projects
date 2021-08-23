#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:21:13 2021

@author: lockiemichalski
"""
##############################################################################

import pandas as pd
import numpy as np
import re
from re import sub
import multiprocessing
from unidecode import unidecode
from pathlib import Path
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from time import time 
from collections import defaultdict
import logging  # Setting up the loggings to monitor gensim
from gensim.parsing.porter import PorterStemmer
import os
import contractions 
import pickle
import itertools
from functools import partial, reduce                                                                              
from nltk.util import ngrams

##############################################################################
'''word2vec reddit daily discussion'''

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
p = PorterStemmer() 

search_dir = '' #directory where reddit pickle files are saved
paths = sorted(Path(search_dir).iterdir(), key=os.path.getmtime)
files = [os.path.join(search_dir, f) for f in paths] # add path to each file
files = [file for file in files if file.endswith('.pkl')]
files.reverse()

def pre_process_text(dd_dict, df_cmc, idx, dd):
    
    dd_df = pd.concat(dd_dict.values(), ignore_index=True) #concat the dictionary to a df
    dd_df = dd_df.drop_duplicates(subset=['comment']) #drop duplicate row comments
    dd_df_comments = dd_df['comment'] #pd series of comment column
    dd_df_comments = dd_df_comments.apply(lambda x: contractions.fix(x)) #fix contractions e.g., 'don't' -> 'do not'
    dd_df_comments = dd_df_comments.apply(lambda x: re.sub(r'[^\w\s]', ' ',x)) #remove punctuation 
    dd_df_comments = dd_df_comments.apply(lambda x: nltk.word_tokenize(x)) #tokenize the sentence -> list of words
    dd_df_comments = dd_df_comments.reset_index() #reset index to do explode - save looping over to replace string
    dd_df_comments = dd_df_comments.assign(var1=dd_df_comments['comment'].str.split(',')).explode('comment')
    dd_dict = df_cmc[['symbol', 'name_lower']].set_index('symbol').to_dict() #dict to replace words in column from
    dd_df_comments['comment'] = dd_df_comments['comment'].replace(dd_dict['name_lower']) #replace cap symbol -> BTC to bitcoin etc. 
    dd_df_comments = (dd_df_comments.groupby(['index']).agg({'comment': lambda x: ' '.join(x.astype(str))})) #reverse the explode
    dd_df_comments = dd_df_comments.apply(lambda x: [w.lower() for w in x]) #lowercase words in lists 
    dd_df_comments = dd_df_comments['comment'].apply(lambda x: nltk.word_tokenize(x))
    dd_df_comments = dd_df_comments.apply(lambda x: [word for word in x if word.isalpha()]) #remove numbers
    dd_df_comments = dd_df_comments.apply(lambda x: [w for w in x if w not in set(stopwords.words('english'))]) #remove stop words
    dd_df_comments_stem = dd_df_comments.apply(lambda x: [p.stem_sentence(w) for w in x]) #stem words
    
    print('Processed {}, {}'.format(str(idx),dd))
    del dd_dict
    
    return dd_df_comments, dd_df_comments_stem

##############################################################################
df_sent_dict = {}
df_sent_stem_dict = {}

for idx, dd in enumerate(files):
    df_dd = pd.read_pickle(dd) #read in the pickle file
    print('Read in, idx {}, {}'.format(str(idx), dd))
    dd_c, dd_c_s = pre_process_text(df_dd, df, idx, dd)
    df_sent_dict[dd.split("Crypto/",1)[1] ] = dd_c
    df_sent_stem_dict[dd.split("Crypto/",1)[1] ] =  dd_c_s 
    del dd_c, dd_c_s, df_dd

output = open('dd_sentences_dict'+'.pkl', 'wb') 
pickle.dump(df_sent_dict, output)

output = open('dd_sentences_stem_dict'+'.pkl', 'wb') 
pickle.dump(df_sent_stem_dict, output)

sent_df_clean = pd.concat(df_sent_dict.values(), ignore_index=True)

##############################################################################
'''Function to get n grams of each sentence in df'''

def n_gram_sentences(sentences_df):
    
    df_ngrams_result = pd.DataFrame() #final n_gram df to output
    
    for index, sentence in enumerate(sent_df_clean):
        if (len(sentence) == 1) or (len(sentence) == 0):
            continue
        
        bigrams_nltk = ngrams(sentence,2) #generator of bigram
        bigrams_nltk = list(bigrams_nltk) #turn generator into list
        trigrams_nltk = ngrams(sentence,3) #generator of trigram
        trigrams_nltk = list(trigrams_nltk) #turn generator into list
        quadgrams_nltk = ngrams(sentence,4) #generator of quadgram
        quadgrams_nltk = list(quadgrams_nltk) #turn generator into list
            
        #list of bi, tri, and quad gram sentences
        n_gram_list = [bigrams_nltk, trigrams_nltk, quadgrams_nltk]
        
        len_tri = len(trigrams_nltk)
        len_quad = len(quadgrams_nltk)
        
        #dict of dataframe for the n_grams -> 2,3,4
        n_gram_dict = {}
        
        for idx, sentence_ngram in enumerate(n_gram_list):
            df_empty = pd.DataFrame() #empty df to store looped sentenceds
            
            '''Sentence 2 strings''' 
            if (len_quad == 0) and (len_tri == 0):
                for sent in sentence_ngram:
    
                    first_word = sent[0] #first word of each sentence
                    n_gram = '_'.join(sent) #create the n_gram
                    
                    '''Bigram'''
                    if idx==0:
                        df_ngram = pd.DataFrame({'first_word':[first_word],
                             'bi_gram_word':[n_gram]})

                    df_empty = df_empty.append(df_ngram) 
                    df_empty.reset_index(drop=True, inplace=True)
                    n_gram_dict[idx]=df_empty
            
            '''Sentence 3 strings''' 
            if (len_quad == 0) and (len_tri > 0):
                for sent in sentence_ngram:
    
                    first_word = sent[0] #first word of each sentence
                    n_gram = '_'.join(sent) #create the n_gram
                    
                    '''Bigram'''
                    if idx==0:
                        df_ngram = pd.DataFrame({'first_word':[first_word],
                             'bi_gram_word':[n_gram]})
                    '''Trigram'''  
                    if idx==1:
                        df_ngram = pd.DataFrame({'first_word':[first_word],
                             'tri_gram_word':[n_gram]})
                    
                    df_empty = df_empty.append(df_ngram) 
                    df_empty.reset_index(drop=True, inplace=True)
                    n_gram_dict[idx]=df_empty
            
            '''Sentence longer than 4 strings''' 
            if (len_quad > 0) and (len_tri > 0):
                for sent in sentence_ngram:
    
                    first_word = sent[0] #first word of each sentence
                    n_gram = '_'.join(sent) #create the n_gram
                    
                    '''Bigram'''
                    if idx==0:
                        df_ngram = pd.DataFrame({'first_word':[first_word],
                             'bi_gram_word':[n_gram]})
                    '''Trigram'''  
                    if idx==1:
                        df_ngram = pd.DataFrame({'first_word':[first_word],
                             'tri_gram_word':[n_gram]})
                    '''Quadgram'''
                    if idx==2:
                        df_ngram = pd.DataFrame({'first_word':[first_word],
                             'quad_gram_word':[n_gram]})
                    
                    df_empty = df_empty.append(df_ngram) 
                    df_empty.reset_index(drop=True, inplace=True)
                    n_gram_dict[idx]=df_empty
                
            '''join strings together in sentence, with first string, 
            n_gram string, second string, second n_gram string'''   
            
            my_reduce = partial(pd.merge, on='first_word', how='outer')                                                              
            n_gram_df = reduce(my_reduce, n_gram_dict.values())   
           
            '''Replace np.nan with something that does not appear in strings,
            cannot make nan, as nan might appear in the strings. Make a string 
            of a float - '1234'
            '''
        
            n_gram_df = n_gram_df.replace(np.nan, '1234', regex=True)
        
        
        '''Final df with sentence with string >= 4'''
        if (len_quad > 0) and (len_tri > 0):
            '''Turn the columns in df into list of bigrams + orginal words'''
            bi_gram_sentence = [x for x in itertools.chain.\
            from_iterable(itertools.zip_longest(
                list(n_gram_df[n_gram_df.columns[0]]),
                list(n_gram_df[n_gram_df.columns[1]]))) if x]
            #remove 1234 in the list
            bi_gram_sentence = [x for x in bi_gram_sentence if x != '1234']
            # last string in sentence is removed. Add back in
            bi_gram_sentence.append(sentence[-1]) 
            
            '''Turn the columns in df into list of trigrams + orginal words'''
            tri_gram_sentence = [x for x in itertools.chain.\
            from_iterable(itertools.zip_longest(
                list(n_gram_df[n_gram_df.columns[0]]),
                list(n_gram_df[n_gram_df.columns[2]]))) if x]
            #remove 1234 in the list
            tri_gram_sentence = [x for x in tri_gram_sentence if x !='1234']
            # last string in sentence is removed. Add back in
            tri_gram_sentence.append(sentence[-1])
    
            '''Turn the columns in df into list of quadgrams + orginal words'''
            quad_gram_sentence = [x for x in itertools.chain.\
            from_iterable(itertools.zip_longest(
                list(n_gram_df[n_gram_df.columns[0]]),
                list(n_gram_df[n_gram_df.columns[3]]))) if x]
            #remove 1234 in the list
            quad_gram_sentence = [x for x in quad_gram_sentence if x !='1234']
            #last string in sentence is removed. Add back in
            quad_gram_sentence.append(sentence[-1])
    
            df_ngram = pd.DataFrame({'original':[sentence],
                                     'bi_gram':[bi_gram_sentence],
                                     'tri_gram':[tri_gram_sentence],
                                     'quad_gram':[quad_gram_sentence]})
              
            df_ngrams_result = df_ngrams_result.append(df_ngram) 
        
        print('Completed sentence idx {}'.format(index))
        print(sentence)
        
        '''Final df with sentence with string = 3'''
        if (len_quad == 0) and (len_tri > 0):
            '''Turn the columns in df into list of bigrams + orginal words'''
            bi_gram_sentence = [x for x in itertools.chain.\
            from_iterable(itertools.zip_longest(
                list(n_gram_df[n_gram_df.columns[0]]),
                list(n_gram_df[n_gram_df.columns[1]]))) if x]
            #remove 1234 in the list
            bi_gram_sentence = [x for x in bi_gram_sentence if x != '1234']
            # last string in sentence is removed. Add back in
            bi_gram_sentence.append(sentence[-1]) 
            
            '''Turn the columns in df into list of trigrams + orginal words'''
            tri_gram_sentence = [x for x in itertools.chain.\
            from_iterable(itertools.zip_longest(
                list(n_gram_df[n_gram_df.columns[0]]),
                list(n_gram_df[n_gram_df.columns[2]]))) if x]
            #remove 1234 in the list
            tri_gram_sentence = [x for x in tri_gram_sentence if x !='1234']
            # last string in sentence is removed. Add back in
            tri_gram_sentence.append(sentence[-1])
    
            df_ngram = pd.DataFrame({'original':[sentence],
                                     'bi_gram':[bi_gram_sentence],
                                     'tri_gram':[tri_gram_sentence]})
              
            df_ngrams_result = df_ngrams_result.append(df_ngram) 
            
        '''Final df with sentence with string = 2'''
        if (len_quad == 0) and (len_tri == 0):
            '''Turn the columns in df into list of bigrams + orginal words'''
            bi_gram_sentence = [x for x in itertools.chain.\
            from_iterable(itertools.zip_longest(
                list(n_gram_df[n_gram_df.columns[0]]),
                list(n_gram_df[n_gram_df.columns[1]]))) if x]
            #remove 1234 in the list
            bi_gram_sentence = [x for x in bi_gram_sentence if x != '1234']
            # last string in sentence is removed. Add back in
            bi_gram_sentence.append(sentence[-1]) 

            df_ngram = pd.DataFrame({'original':[sentence],
                                     'bi_gram':[bi_gram_sentence]})
              
            df_ngrams_result = df_ngrams_result.append(df_ngram) 
            
        return df_ngrams_result
    
 ##############################################################################

def bigrams_gensim(sentences_df, minimum_count=3):
    bigram_phrases = Phrases(sentences_df, min_count=minimum_count, progress_per=10000)
    bigram_sentences = []
    for unigram_sentence in sentences_df:                
        bigram_sentence = u' '.join(bigram_phrases[unigram_sentence])
        bigram_sentences.append(bigram_sentence)
        
    df_bigram_sent = pd.Series(bigram_sentences)     
    df_bigram_sent = df_bigram_sent.apply(lambda x: nltk.word_tokenize(x)) #tokenize the sentence -> list of words

    return df_bigram_sent 

def trigrams_gensim(bigram_sentences, minimum_count=3):
    trigram_phrases = Phrases(bigram_sentences, min_count=minimum_count, progress_per=10000)
    trigram_sentences = []
    for bigram_sentence in bigram_sentences:                
        trigram_sentence = u' '.join(trigram_phrases[bigram_sentence])
        trigram_sentences.append(trigram_sentence)
        
    df_trigram_sent = pd.Series(trigram_sentences)     
    df_trigram_sent = df_trigram_sent.apply(lambda x: nltk.word_tokenize(x)) #tokenize the sentence -> list of words

    return df_trigram_sent 

bigram = bigrams_gensim(sent_df_clean, minimum_count=3)
trigram = trigrams_gensim(bigram, minimum_count=3)

##############################################################################
'''Creating bigrams'''
#to create the bigrams
bigram_model = Phrases(df_sentences, min_count=3, progress_per=10000)
#apply the trained model to a sentence
bigram_sentences = []
for unigram_sentence in df_sentences:                
    bigram_sentence = u' '.join(bigram_model[unigram_sentence])
    bigram_sentences.append(bigram_sentence)
    
df_bigram_sent = pd.Series(bigram_sentences)     
df_bigram_sent = df_bigram_sent.apply(lambda x: nltk.word_tokenize(x)) #tokenize the sentence -> list of words

'''Creating trigrams'''
#get a trigram model out of the bigram
trigram_model = Phrases(df_bigram_sent)
trigram_sentences = []
for unigram_sentence in df_bigram_sent:                
    trigram_sentence = u' '.join(trigram_model[unigram_sentence])
    trigram_sentences.append(trigram_sentence)
    
df_trigram_sent = pd.Series(trigram_sentences)     
df_trigram_sent = df_trigram_sent.apply(lambda x: nltk.word_tokenize(x)) #tokenize the sentence -> list of words

'''Sanity check'''
word_freq = defaultdict(int)
for sent in df_trigram_sent:
    for i in sent:
        word_freq[i] += 1
len(word_freq)

##############################################################################
'''Word2Vec'''

'''Why seperate the training of the model in 3 steps:
Prefer to separate the training in 3 distinctive steps for clarity and monitoring.

1. Word2Vec():
    In this first step, I set up the parameters of the model one-by-one.
    I do not supply the parameter sentences, and therefore leave the model uninitialized, purposefully.

2. .build_vocab():
    Here it builds the vocabulary from a sequence of sentences and thus initialized the model.
    With the loggings, I can follow the progress and even more important, the effect of min_count and sample on the word corpus. I noticed that these two parameters, and in particular sample, have a great influence over the performance of a model. Displaying both allows for a more accurate and an easier management of their influence.

3. .train():
    Finally, trains the model.
    The loggings here are mainly useful for monitoring, making sure that no threads are executed instantaneously.'''

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

'''
The parameters:
    
1.  min_count = int - Ignores all words with total absolute frequency lower 
    than this - (2, 100)
    
2.  window = int - The maximum distance between the current and predicted word 
    within a sentence. 
    E.g. window words on the left and window words on the left of our target - (2, 10)
    
3.  vector_size = int - Dimensionality of the feature vectors. - (50, 300)

4.  sample = float - The threshold for configuring which higher-frequency words are 
    randomly downsampled. Highly influencial. - (0, 1e-5)
    
5.  alpha = float - The initial learning rate - (0.01, 0.05)

6.  min_alpha = float - Learning rate will linearly drop to min_alpha as training progresses. 
    To set it: alpha - (min_alpha * epochs) ~ 0.00
    
7.  negative = int - If > 0, negative sampling will be used, the int for negative 
    specifies how many "noise words" should be drown. If set to 0, no negative 
    sampling is used. - (5, 20)
    
8.  workers = int - Use these many worker threads to train the model 
    (=faster training with multicore machines)'''

'''Step 1 - Word2Vec'''

w2v_model_hs = Word2Vec(min_count=1,
                     window=3,
                     vector_size=200,
                     sample=1e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)

'''Step 2 - Building the Vocabulary Table:
Word2Vec requires us to build the vocabulary table (simply digesting all the 
words and filtering out the unique words, and doing some basic counts on them):'''

start = time()
w2v_model_hs.build_vocab(df_ngrams_result['bi_gram'], progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - start) / 60, 2)))


'''Step 3 - Training of the model:
Parameters of the training:

total_examples = int - Count of sentences;
epochs = int - Number of iterations (epochs) over the corpus - [10, 20, 30]'''

start = time()
w2v_model_hs.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - start) / 60, 2)))


'''Exploring the model
Most similar to:
Here, we will ask our model to find the word most similar to bitcoin'''

w2v_model.wv.most_similar("bitcoin")  
list(w2v_model.wv.vocab.keys())
words = w2v_model.wv.key_to_index
len(w2v_model_sg.wv)

#w2v_model.wv.most_similar(positive=["xrp"]) #ripple
#w2v_model.wv.most_similar("musk") #elon musk

##############################################################################
# Store just the words + their trained embeddings.
word_vectors = w2v_model.wv
word_vectors.save("word2vec.wordvectors")
# Load back with memory-mapping = read-only, shared across processes.
wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
vector = wv['bitcoin']  # Get numpy vector of a word

'''Cluster similar words together'''
model = KMeans(n_clusters=10, max_iter=10000, random_state=True, n_init=100).fit(X=word_vectors.vectors.astype('double'))
labels = model.labels_
centroids = model.cluster_centers_

#have a look at different clusters of words 
word_vectors.similar_by_vector(model.cluster_centers_[0], topn=20, restrict_vocab=None)
word_vectors.similar_by_vector(model.cluster_centers_[1], topn=20, restrict_vocab=None)
word_vectors.similar_by_vector(model.cluster_centers_[2], topn=20, restrict_vocab=None)
word_vectors.similar_by_vector(model.cluster_centers_[3], topn=100, restrict_vocab=None)
word_vectors.similar_by_vector(model.cluster_centers_[4], topn=20, restrict_vocab=None)
word_vectors.similar_by_vector(model.cluster_centers_[5], topn=20, restrict_vocab=None)
word_vectors.similar_by_vector(model.cluster_centers_[6], topn=20, restrict_vocab=None)
word_vectors.similar_by_vector(model.cluster_centers_[7], topn=20, restrict_vocab=None)
word_vectors.similar_by_vector(model.cluster_centers_[8], topn=20, restrict_vocab=None)
word_vectors.similar_by_vector(model.cluster_centers_[9], topn=20, restrict_vocab=None)

##############################################################################

