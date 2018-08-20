import pandas as pd
import pickle as p
import word2vec
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from modelx import modelx

def get_x(csv_path,
          col,
          max_length,
         embedding_file,
          x_file):
    token_length = max_length
    w2v = word2vec.load(embedding_file)
    reserve_voc = {token:indict+1 for indict, token in enumerate(w2v.vocab)}
    
    def text2idx(text):
        idx= []
        for token in text.split():
            if token in reserve_voc.keys():
                idx.append(reserve_voc[token])
        return idx
    
    train_x = pd.read_csv(csv_path, usecols=[col])[col].apply(text2idx)
    data = pad_sequences(train_x, maxlen=token_length)
    embedding_matrix = np.r_[np.zeros(shape=(1,w2v.vectors.shape[1])), w2v.vectors]
    x = modelx(embedding_matrix, token_length, data)
    with open(x_file, 'wb') as f:
        p.dump(x, f)