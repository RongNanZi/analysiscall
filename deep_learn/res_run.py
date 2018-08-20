import os
import pandas as pd
import pickle as p
import numpy as np
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import *

import tensorflow as tf
import sys
sys.path.append('../data')
from modelx import modelx
# ========system setting======
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

#=========data load======
data_path = '../data/'
        
csv_data = pd.read_csv(data_path+'call_reason.csv',usecols=['label'])
data_x = p.load(open(data_path+'token_20Kbaike.x', 'rb'))
from models import *
from sklearn.model_selection import train_test_split


train_x, test_x, train_y, test_y = train_test_split(data_x.data, csv_data['label'], test_size=0.2)

#=====callback object set
import logging
logging.basicConfig(filename='train_all.log', level=logging.INFO)
logging.info('========================================================\n ')

tf_board_op = TensorBoard(log_dir='./logs/ResNet/baike',
                            write_graph=False,
                            write_images=True,
                            embeddings_freq=0, embeddings_metadata=None)
model_save_dir = './model_file/ResNet/'

if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
tf_save_op = ModelCheckpoint(model_save_dir+'token_{val_acc:.4f}.hdf5',
                                             monitor='val_acc',
                                             verbose=1,
                                             save_best_only=True,
                                             save_weights_only=False,
                                             mode='max',
                                             period=1)
es_op = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max')

def get_session():
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    #tf_config.gpu_options.allocator_type ='BFC'
    return tf.Session(config=tf_config)
KTF.set_session(get_session())

from keras.engine import Input
from keras.models import Model

def get_model(embedding_matrix,
              max_length,
             n_class):
    inputs = Input(shape = (max_length,))

    embedding_vec = Embedding( input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
        weights= [embedding_matrix],     trainable=True)(inputs)

    logging.info('the model is use res  net, \n \t ngrams = [64, 128, 128],\n \
    \t level : token  \n \t embedding : static  \n \t max_length : 150 \n' )
    logist = Resnet(embedding_vec, n_class=n_class, l2_a=0)
    model = Model(inputs=inputs, outputs=logist)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = get_model(data_x.embedding_matrix, data_x.max_length, 4)
history = model.fit(x = train_x,
                    y = train_y, 
                                     batch_size=256,
                                     epochs=100,
                                     verbose=1, 
                                     callbacks=[tf_board_op, tf_save_op, es_op],
                                     validation_data = (test_x, test_y),
                                     shuffle=True)
logging.info(history.history)
