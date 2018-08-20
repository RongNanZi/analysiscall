
from gensim.models import word2vec
import pandas as pd
import pickle as p
from sklearn.feature_extraction.text import TfidfVectorizer

csv = pd.read_csv('call_reason.csv', usecols=['token'])
#csv['text'] = csv['text'].apply(lambda x: ' '.join([item for item in x]))

def get_voc_embedding(data, col, voc_name, embed_name):
            vec = TfidfVectorizer(max_df=0.85, min_df= 3,
                                   smooth_idf=1, sublinear_tf=1)
            vec.fit(data[col])
            voc = vec.vocabulary_
            p.dump(voc, open(voc_name, 'wb'))
                          
            sentences = data[col].apply(lambda x:[item for item in x.split() if item in voc.keys()]).values  
            model = word2vec.Word2Vec(sentences,min_count =0, hs=1,  window=5, size=256, max_vocab_size =None)
                                        
            print('the voc length is {}, the embedding size is {}'.format(len(voc), model.wv.vectors.shape))
            model.save(embed_name)
                                                        
get_voc_embedding(csv, 'token', 'token.voc', 'token_em.model')
