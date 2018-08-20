from  sklearn.feature_extraction.text import TfidfVectorizer
import pickle as p
import pandas as pd

data = pd.read_csv('call_reason.csv', usecols=['text','token'])

def save(df, col, f_name, ngram=None):
    if ngram == None:
        vec = TfidfVectorizer(max_df=0.9, min_df= 3,
                            smooth_idf=1, sublinear_tf=1)
    else: vec = TfidfVectorizer(ngram_range=ngram, max_df=0.9, min_df= 3,
                            smooth_idf=1, sublinear_tf=1)
    t_tfidf = vec.fit_transform(df[col])
    with open(f_name, 'wb') as f:
        p.dump(t_tfidf, f)

save(data, 'text', 'word.tfidf', (1,2))
save(data, 'token', 'token.tfidf')

