from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential, Model, Input
from keras.layers import LSTM, TimeDistributed , Dense, Dropout, Bidirectional, Concatenate
import data_model_jp as datamodel
from sklearn.model_selection import train_test_split
import numpy as np


# In[2]:


def get_ed_model(hidden_size,  n_x_features ,  n_y_features):
    
    #define_model
    ##encoder
    encoder_in = Input(shape=(None,n_x_features))
    encoder_out, state_h, state_c = LSTM(hidden_size, return_state=True)(encoder_in)
    
    context_vector = [state_h, state_c]
    
    ##decoder
    decoder_in = Input(shape=(None,n_y_features))
    decoder_out, _,_ = LSTM(hidden_size, return_state=True, return_sequences=True)(decoder_in, initial_state=context_vector)
    
    ##softmax
    softmax = Dense(n_y_features, activation='softmax')
    softmax_out = softmax(decoder_out)
    
    ##encoder-decoder moder
    ed_model = Model([encoder_in, decoder_in],softmax_out)
    ed_model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    '''
    inference model    
    '''
    #define ed inference model
    ##encoder
    encoder_inf = Model(encoder_in, context_vector)
    
    ##decoder - one timestep model
    decoder_inf_in = Input(shape=(None,n_y_features))
    decoder_state_in = [Input(shape=(hidden_size,)), Input(shape=(hidden_size,))]
    decoder_inf_out, decoder_inf_h, decoder_inf_c = LSTM(hidden_size, return_state=True, return_sequences=True)(decoder_inf_in,initial_state=decoder_state_in)
    
    decoder_state_out = [decoder_inf_h, decoder_inf_c]
    
    softmax_out = softmax(decoder_inf_out)
    
    
    ## decoder-model-one-timestep
    decoder_inf = Model([decoder_inf_in] + decoder_state_in , [softmax_out] + decoder_state_out)    
    
    return ed_model, encoder_inf, decoder_inf
    


# In[3]:


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def decode_to_sent(x,idx_to_item, delim, pad, in_level):
    if in_level == 'w_ch':
        sub_item_len = len(idx_to_item)
        sub_items_list = []
        for item in x:
            sub_items = list(chunks(item, sub_item_len))
            sub_items = [np.where(sub_item == 1)[0] for sub_item in sub_items]
            sub_items = [idx_to_item[sub_item[0]] if len(sub_item)>0 else pad for sub_item in sub_items]
            sub_items_list.append(''.join(sub_items))
        items = delim.join(sub_items_list)
        items = items.rstrip()
        items = items.lstrip()
        return items
    else:
        items = [np.where(val == 1)[0] for val in x]
        items = [idx_to_item[item[0]] if len(item)>0 else pad for item in items]
        items = delim.join(items)
        items = items.rstrip()
        items = items.lstrip()
        return items


# In[4]:


def safe_division(n, d):
    return n / d if d>0 else 0


# In[5]:


def infer(encoder_inf, decoder_inf, x, x_idx_to_item, y_item_to_idx, n_x_features, n_y_features, start_delim, end_delim, max_timesteps=50):
    output = []
    y_idx_to_item = {y_item_to_idx[key]:key for key in y_item_to_idx.keys()}
    
    decoder_prev_state = encoder_inf.predict(x.reshape(1,len(x),n_x_features))
    
    out_item = start_delim
    
    
    for i in range(max_timesteps):
        hot_encoded_input = np.zeros(shape=(1, 1, n_y_features))
        hot_encoded_input[0,0,y_item_to_idx[out_item]] = 1
        
        decoder_out, decoder_h, decoder_c = decoder_inf.predict([hot_encoded_input]+decoder_prev_state)
        decoder_prev_state = [decoder_h, decoder_c]
        
        out_item = y_idx_to_item[np.argmax(decoder_out)]
        output.append(out_item)

        if out_item == end_delim:
            break
    
    return output


# In[6]:


def train_test(x_filename, y_filename, hidden_size=32, n_epochs=10, mini_batch_size=32, val_split=0.2,
               portion=1, test_size=0.2, in_level_x = 'w', in_level_y = 'w' ):
        
    
    #get data
    data , dicts, folder = datamodel.prepare_dataset(x_filename, y_filename, 
                                             portion = portion, test_size=0.2,
                                             in_level_x = in_level_x, in_level_y = in_level_y)

    x_enc_tr, x_enc_te = data[0][0], data[0][1]
    y_dec_in_tr, _ = data[1][0], data[1][1]
    y_dec_out_tr, y_dec_out_te = data[2][0], data[2][1]
    x_item_to_idx, y_item_to_idx = dicts[0], dicts[1]
    x_idx_to_item = {x_item_to_idx[key]:key for key in x_item_to_idx.keys()}
    y_idx_to_item = {y_item_to_idx[key]:key for key in y_item_to_idx.keys()}

    print('x: ', x_enc_tr.shape, x_enc_te.shape)
    print('y: ', y_dec_out_tr.shape, y_dec_out_te.shape)
    #settings
    start_delim, end_delim = '', ''
    if 'ch' in in_level_y :
        start_delim, end_delim = '\t', '\n'
    else:
        start_delim, end_delim = 'SOS', 'EOS'
        
    n_x_features, n_y_features = 0, 0
    if in_level_x == 'ch' or in_level_x == 'w':
        n_x_features = len(x_item_to_idx)
    else:
        n_x_features = len(x_item_to_idx) * len(x_enc_tr[0])
        
    if in_level_y == 'ch' or in_level_y == 'w':
        n_y_features = len(y_item_to_idx)
    else:
        n_y_features = len(y_item_to_idx) * len(y_dec_in_tr[0])
        
    
    #get model
    ed_model, encoder_inf, decoder_inf = get_ed_model(32, n_x_features, n_y_features)
    print(ed_model.summary())
    
    #train model
    ed_model.fit(x=[x_enc_tr,y_dec_in_tr],  y=y_dec_out_tr, epochs = n_epochs,batch_size = mini_batch_size, validation_split = val_split, verbose=2)
    
    #test model
    d_tp, d_tn, d_fp, d_fn = 0, 0, 0, 0 #detection
    c_tp, c_tn, c_fp, c_fn = 0, 0, 0, 0 #correction
    
    for i, x in enumerate(x_enc_te):
        output = infer(encoder_inf,decoder_inf, x, x_idx_to_item, y_item_to_idx, n_x_features, n_y_features, start_delim, end_delim,len(x))
         
        #evaluate
        delim = '' if in_level_x=='ch' else ' '
        _input = decode_to_sent(x, x_idx_to_item, delim, '', in_level_x)
        exp_output = decode_to_sent(y_dec_out_te[i], y_idx_to_item, delim, '',in_level_y)
        pred_output = delim.join(output)
        pred_output = pred_output.rstrip()
        pred_output = pred_output.lstrip()
        print(_input)
        print(exp_output)
        print(pred_output)
        
        if (_input == exp_output): # no error
            if(exp_output == pred_output):
                d_tn += 1 #no error, not detected an error
                c_tn += 1 #should not be corrected, not corrected
            else:
                d_fp += 1 #no error, but detected an error
                c_fp += 1 #should not be corrected, but corrected
        else: # error
            if(exp_output == pred_output): 
                d_tp += 1 #error, detected an error
                c_tp += 1 #should be corrected, corrected accurately
            else:
                d_fn += 1 #error, but not detected an error
                c_fn += 1 #should be corrected, but not corrected
#                 if(pred_output == _input):
#                     d_fn += 1 #error, but not detected an error
#                     c_fn += 1 #should be corrected, but not corrected
#                 else:
#                     d_tp += 1 #error, detected an error ?
#                     c_fn += 1 #should be corrected, but not corrected accurately
    
    with open(folder+'\\results.txt', 'w+', encoding="utf8") as file:
        d_recall = safe_division(d_tp,d_tp + d_fp)
        d_prec = safe_division(d_tp,d_tn + d_tp)
        c_recall = safe_division(c_tp,c_tp + c_fp)
        c_prec = safe_division(c_tp,c_tn + c_tp)
        
        file.write('Detection TP {} \n'.format(d_tp))
        file.write('Detection TN {} \n'.format(d_tn))
        file.write('Detection FP {} \n'.format(d_fp))
        file.write('Detection FN {} \n'.format(d_fn))
        file.write('Detection Accuracy: {} \n'.format(safe_division((d_tn + d_tp), (d_tn + d_tp + d_fn + d_fp))))
        file.write('Detection Recall: {} \n'.format(d_recall))
        file.write('Detection Precision: {} \n'.format(d_prec))
        file.write('Detection F1 {} \n'.format(safe_division((2 * d_recall * d_prec) , (d_recall + d_prec))))
        file.write('{} \n'.format('='*20))
        file.write('Correction TP {} \n'.format(c_tp))
        file.write('Correction TN {} \n'.format(c_tn))
        file.write('Correction FP {} \n'.format(c_fp))
        file.write('Correction FN {} \n'.format(c_fn))
        file.write('Correction Accuracy: {} \n'.format(safe_division((c_tn + c_tp), (c_tn + c_tp + c_fn + c_fp))))
        file.write('Correction Recall: {} \n'.format(c_recall))
        file.write('Correction Precision: {} \n'.format(c_prec))
        file.write('Correction F1 {} \n'.format(safe_division((2 * c_recall * c_prec), (c_recall + c_prec))))
        file.close()
    
    # serialize model to JSON
    model_json = ed_model.to_json()
    with open(folder+"\\model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    ed_model.save_weights("model.h5")
    print(folder+"\\Saved model to disk")

train_test('sentences_bad','sentences_good',portion=1, n_epochs=10, in_level_x='w_ch', in_level_y='w')





