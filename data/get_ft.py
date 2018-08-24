import pandas as pd
from sklearn.model_selection import train_test_split

csv = pd.read_csv('call_reason.csv', usecols=['label', 'token'])
train, test = train_test_split(csv, test_size=0.2)

def save(data, f_n):
        for i in data.index:
                with open(f_n, 'a')as f:
                        line = '__label__{},{}\n'.format(data.loc[i]['label'], data.loc[i]['token'])
                        f.write(line)
save(train, 'ft_train.txt')
save(test, 'ft_test.txt')
