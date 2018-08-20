import pandas as pd
import pickle as p
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from modelx import modelx
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def get_gensim_embedding(data, col, embed_name):
    if os.path.exists(embed_name):
        return word2vec.Word2Vec.load(embed_name)
    
    if col != 'token':
        sentences = data[col].values
        model = word2vec.Word2Vec(sentences, hs=1,  window=5, size=128, max_vocab_size =None)
        
    else:
        #get vocabulary by tfidf
        vec = TfidfVectorizer(max_df=0.9, min_df= 3,
                                smooth_idf=1, sublinear_tf=1)
        vec.fit(data[col])
        voc = vec.vocabulary_
        sentences = data[col].apply(lambda x:[item for item in x.split() if item in voc.keys()]).values  
        model = word2vec.Word2Vec(sentences,min_count = 0, hs=1,  window=5, size=128, max_vocab_size =None)
    print('the embedding size is {}\n save it!\n'.format(model.wv.vectors.shape))
    model.save(embed_name)
    return model

def get_gensim_text_x(csv_path,
          col,
          max_length,
         embedding_file,
          x_file):
    token_length = max_length
    csv = pd.read_csv(csv_path, usecols=[col])
    w2v = get_gensim_embedding(csv, col, embedding_file)
    voc = w2v.wv.vocab
    
    def text2idx(text):
        idx= []
        for token in text:
            if token in voc.keys():
                idx.append(voc[token].index+1)
        return idx
    
    train_x = pd.read_csv(csv_path, usecols=[col])[col].apply(text2idx)
    #w2v.wv.vocab['unk'].index = 126
    embedding_matrix = np.r_[np.zeros(shape=(1,w2v.wv.vectors.shape[1])), w2v.wv.vectors]
    data = pad_sequences(train_x, maxlen=token_length, value=0)
    
    x = modelx(embedding_matrix, token_length, data)
    with open(x_file, 'wb') as f:
        p.dump(x, f)
        
def get_gensim_token_x(csv_path,
          col,
          max_length,
         embedding_file,
          x_file):
    token_length = max_length
    csv = pd.read_csv(csv_path, usecols=[col])
    w2v = get_gensim_embedding(csv, col, embedding_file)
    voc = w2v.wv.vocab
    
    def text2idx(text):
        idx= []
        for token in text.split():
            if token in voc.keys():
                idx.append(voc[token].index)
        return idx
    
    train_x = pd.read_csv(csv_path, usecols=[col])[col].apply(text2idx)
    #w2v.wv.vocab['unk'].index = 126
    embedding_matrix = w2v.wv.vectors
    data = pad_sequences(train_x, maxlen=token_length, value=voc['unk'].index)
    
    x = modelx(embedding_matrix, token_length, data)
    with open(x_file, 'wb') as f:
        p.dump(x, f)
 