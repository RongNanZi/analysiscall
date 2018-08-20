from sklearn.svm import LinearSVC 
import pickle as p
import logging
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, f1_score 

logging.basicConfig(filename='svm.log', level=logging.INFO)
data_path = '../data/'

word_tfidf = p.load(open(data_path+'word.tfidf', 'rb'))
token_tfidf = p.load(open(data_path+'token.tfidf', 'rb'))

# because the nums of calss 4,5 noly 1,2, so just drop to 0 class
y = pd.read_csv(data_path+'call_reason.csv', usecols=['label'])['label'].apply(lambda x: x if x < 4 else 0)

def eval(x, y, name):
        clf = LinearSVC()
        c_v_score = cross_validate(clf, x, y, cv=5, scoring=['accuracy', 'f1_macro']) 
        logging.info('===========the {} svc result===========\n \
                     the cross val score is\n the average rain_acc : {}\t train_f1 : {} \n \
                     the test acc : {}\t f1 : {} \
                     '.format(name, sum(c_v_score['train_accuracy'])/5, sum(c_v_score['train_f1_macro'])/5,\
                             sum(c_v_score['test_accuracy'])/5,sum(c_v_score['test_accuracy'])/5))
eval(word_tfidf, y, 'word_level')
eval(token_tfidf, y, 'token_level')

