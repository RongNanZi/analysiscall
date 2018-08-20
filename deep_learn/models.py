from keras.layers import Embedding, Conv1D, Dense, Dropout, Activation, MaxPooling1D
from keras.layers.core import Flatten
from keras import regularizers
from keras.layers import add
from keras.layers.normalization import BatchNormalization
from keras.layers import CuDNNGRU
from keras import regularizers
from kutilities.layers import Attention


from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.core import Dropout
from keras.layers import Concatenate

def res_block(inputs, kernel_size, out_size, l2_a, dropout):

    outputs = inputs
    for i, k_size in enumerate(kernel_size):
        
        outputs = Conv1D(filters = out_size,
                    kernel_size = kernel_size[0],
                    padding = 'same',
                    activation=None)(outputs)
        
        outputs = Activation('relu')(outputs)
        
        outputs = Dropout(dropout)(outputs)
    """   
    output1 = Conv1D(filters = out_size,
                    kernel_size = 1,
                    padding = 'same',
                    kernel_regularizer = regularizers.l2(l2_a),
                    activation=None)(inputs)
    output2 = Conv1D(filters = out_size,
                    kernel_size = kernel_size[0],
                    padding = 'same',
                    kernel_regularizer = regularizers.l2(l2_a),
                    activation=None)(inputs)

    output3 = Concatenate(-1)([output1, output2])
    output4 = Activation('relu')(output3)
    
    
    output7 = Conv1D(filters = out_size,
                    kernel_size = 1,
                    padding = 'same',
                    kernel_regularizer = regularizers.l2(l2_a),
                    activation=None)(output4)
    output8 = Conv1D(filters = out_size,
                    kernel_size = kernel_size[1],
                    padding = 'same',
                    kernel_regularizer = regularizers.l2(l2_a),
                    activation=None)(output4)
    
    output9 = Concatenate(-1)([output7, output8])
    output10 = Activation('relu')(output9)
    """
    

    # downsample by 1*1 filter
    if inputs.shape[-1] != outputs.shape[-1]:
        inputs = Conv1D(filters = int(outputs.shape[-1]), kernel_size=1)(inputs)
    out = add([inputs, outputs])
        
    return out
    
def Resnet(inputs,
           n_class,
           l2_a = 0.0001,
           dropout = 0.5,
           attention = False,
                 kernel_size = [2,3],
                 block_outputs=[64, 128, 128]):
    """
    =======================
    the inputs' shape is [batch_size, max_length, embedding]
    n_class :  the counters of classes
    l2_a  : l2 regularizers
    kernel_size : conv kernel size per res block
    block_outputs : filters size per block
    ======================
    """
    if attention:
            inputs = Attention()(inputs)
    #======N bolcks in model
    for i, out_channle  in enumerate(block_outputs):
        if i == 0:
            m_out = res_block(inputs=inputs, kernel_size=kernel_size, out_size=out_channle, dropout=dropout, l2_a=l2_a)
        else:
            m_out = res_block(inputs=m_out, kernel_size=kernel_size, out_size=out_channle, dropout=dropout, l2_a=l2_a)
        m_out = MaxPooling1D(pool_size = 2)(m_out)
    f_out = Flatten()(m_out)
    f_out = Dense(256, kernel_regularizer=regularizers.l2(l2_a),activation='tanh')(f_out)
    f_out = Dropout(dropout)(f_out)
    f_out = Dense(n_class, activation='softmax')(f_out)
    return f_out




def aug_cnn(inputs, gram_filters, n_class, l2_a=0.00001):
    data_aug = []
    for i, c_conf in enumerate(gram_filters):
        #======= inputs shape is (length, embedding)
        f_l = Conv1D(kernel_size = i+2,
                               filters = gram_filters[i],
                               padding = 'valid',
                               kernel_regularizer=regularizers.l2(l2_a),
                               name='aug_{}st'.format(i+1))(inputs)
        #========= out shape is (length, fiters)
        #f_l = BatchNormalization()(f_l )
        f_l = Activation('tanh')(f_l )
        data_aug.append(GlobalMaxPooling1D()(f_l )) 
        
    concat_data = Concatenate(-1)(data_aug)
    m_l = Dense(256, activation='relu')(concat_data)
    m_l = Dropout(0.5)(m_l)
    logist = Dense(n_class, activation='softmax')(m_l)
    return logist

def tcn_block(inputs,
             filters,
              dilation):
    conv1 = Conv1D(filters, kernel_size=2, padding='causal', dilation_rate=dilation)(inputs)
    conv1 = BatchNormalization()(conv1)
    after_a = Activation('relu')(conv1)
    after_d = Dropout(0.5)(after_a)
    
    conv2 = Conv1D(filters, kernel_size=3, padding='causal', dilation_rate=dilation)(after_d)
    conv2 = BatchNormalization()(conv2)
    after_a = Activation('relu')(conv2)
    after_d = Dropout(0.5)(after_a)
    
    if inputs.shape[-1] != after_d.shape[-1]:
        inputs = Conv1D(filters= int(after_d.shape[-1]), kernel_size=1)(inputs)
        
    return add([inputs, after_d])
        
def tcn(inputs, 
        n_class, 
        channels,
        kernel_size = 2):
    
    for i, filters in enumerate(channels):
        dilation = 2 ** i
        if i == 0:
            m_out = tcn_block(inputs = inputs,  filters=filters, dilation = dilation)
        else:
            m_out = tcn_block(inputs = m_out, filters=filters, dilation = dilation)
    m_out = GlobalMaxPooling1D()(m_out) 
    d_out = Dense(32, activation = 'tanh')(m_out)
    d_out = Dropout(0.5)(d_out)
    logist = Dense(n_class,  activation = 'softmax')(d_out)
    return logist

def mix_cnn_rnn(inputs, n_class, channels, l2_a=0.0001):
    
    data_aug = []
    for i, k_size in enumerate(channels):
        data_aug.append(Conv1D(kernel_size = i+1,
                               filters = k_size,
                               padding = 'same',
                               kernel_regularizer=regularizers.l2(l2_a),
                               activation = 'tanh',
                               name='aug_{}st'.format(i+1))(inputs))
        
    concat_data = Concatenate()(data_aug)
    rnn_result = CuDNNGRU(256,return_sequences=True)(concat_data)
    #rnn_result = Dropout(0.5 )(rnn_result)
    #rnn_result = CuDNNGRU(256,return_sequences=True)(rnn_result)
    after_att = Attention()(rnn_result)
    logist = Dense(n_class, kernel_regularizer=regularizers.l2(l2_a), activation='softmax')(after_att)
    return logist
