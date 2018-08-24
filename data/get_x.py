import pandas as pd
import pickle as p
import word2vec
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from modelx import modelx
import os
from sklearn.feature_extraction.text import TfidfVectorizer


def get_embed(csv, col, embed_file):
        if os.path.exists(embed_file):
                return word2vec.load(embed_file)
            
        def csv2txt(text, voc):
            text = ' '.join([item for item in text.split() if item in voc])
            text += '\n'
            with open('word_token.txt', 'a')as f:
                f.write(text)
        os.system('rm word_token.txt')
        
        vec = TfidfVectorizer(max_df=0.9, min_df= 3,smooth_idf=1, sublinear_tf=1)
        vec.fit(csv[col])
        voc = vec.vocabulary_
        
        csv[col].apply(csv2txt, args=[voc])
        word2vec.word2vec('word_token.txt', embed_file, 256, verbose=1)
        return word2vec.load(embed_file)

def get_x(csv_path,
          col,
          max_length,
         embedding_file,
          x_file):
    token_length = max_length
    csv_data = pd.read_csv(csv_path, usecols=[col])
    w2v = get_embed(csv_data, col, embedding_file)
    reserve_voc = {token:indict+1 for indict, token in enumerate(w2v.vocab)}
    
    def text2idx(text):
        idx= []
        for token in text.split():
            if token in reserve_voc.keys():
                idx.append(reserve_voc[token])
        return idx
    
    train_x = csv_data[col].apply(text2idx)
    data = pad_sequences(train_x, maxlen=max_length)
    embedding_matrix = [np.zeros_like(w2v.vectors[0])] + w2v.vectors
    x = modelx(embedding_matrix, max_length, data)
    with open(x_file, 'wb') as f:
        p.dump(x, f)
