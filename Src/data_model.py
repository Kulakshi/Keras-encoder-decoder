#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from __future__ import division
import numpy as np
from statistics import mean, median
import nltk
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
import os.path
import pickle
from statistics import mean, median


# In[2]:


def get_length_statics(data):
    lengths = [len(record) for record in data]
    return max(lengths), mean(lengths)

def read_sentences(file_path, portion):
    data_file = open(file_path, 'r', encoding="utf8")   
    sentences = data_file.readlines()
    data_file.close()
    
    def clean_sentence(sent):
        sentence = ''
        for ch in sent:
            if u'\u0d80' <= ch <= u'\u0dff' or ch == ' ' or ch.isnumeric():
                sentence = sentence + ch
        return sentence

    sents = []
    for i in range(int(3*len(sentences)/5)):
        sent = sentences[i]
        sent = clean_sentence(sent)
        sents.append(sent)
    return sents[:int(len(sents) * portion)]

def mark_start_end(sentences, start_mark, end_mark):
    sents = []
    for sent in sentences:
        sent = start_mark + sent + end_mark
        sents.append(sent)
    return sents

def get_words(sentences):
    sents = []
    for i, sent in enumerate(sentences):
        words = word_tokenize(sent)
        sents.append(words)
    return sents        
    
def prepare_vocab(words):
    word2idx = {}
    word2idx = {w: i+2 for i, w in enumerate(words)}
    word2idx["$"] = 1#unk
    word2idx["#"] = 0#pad
    return word2idx    

def save_np_array(file_path, nparray):
#     file = open(file_path,"w+")
    np.save(file_path, nparray)


# In[3]:


def prepare_tensor(data, offset=0, is_w_ch=True):

    # convert to hot encoding
    if is_w_ch:
        items = set([sub_item for record in data for item in record for sub_item in item])
        item_to_idx = prepare_vocab(items)

        #define tensor
        batch_size = len(data)
        n_timesteps = int(get_length_statics(data)[1]) #mean
        n_features = len(item_to_idx)

        print(batch_size, n_timesteps, n_timesteps*n_features, n_features)
        
        tensor = np.zeros(shape= (batch_size, n_timesteps, n_timesteps*n_features), dtype='uint8')
        for i in range(batch_size): #n_sents
            for j,item in enumerate(data[i]): #words in sent
                if j >= n_timesteps: break
                chars = []
                for k,sub_item in enumerate(item): #char per word
                    if j > offset-1:
                        tensor[i, j-offset, (item_to_idx[sub_item]+(j*n_features))] = 1
    else:
        items = set([item for record in data for item in record])
        item_to_idx = prepare_vocab(items)

        #define tensor
        batch_size = len(data)
        n_timesteps = int(get_length_statics(data)[1]) #mean
        n_features = len(item_to_idx) 
        
        tensor = np.zeros(shape= (batch_size, n_timesteps, n_features), dtype='uint8')

        for i in range(batch_size):
            for j, item in enumerate(data[i]):
                if j >= n_timesteps: break
                if j > offset-1:
                    tensor[i, j-offset, item_to_idx[item]] = 1
    return tensor, item_to_idx      


# In[4]:


def prepare_dataset(x_filename, y_filename, portion = 1, test_size=0.2, in_level_x = 'ch', in_level_y = 'ch'):
    folder = '.\..\Data\\' + x_filename.split('_')[0] +in_level_x+'_'+in_level_y+ '_p' + str(portion) + '_t' + str(test_size)
    
    data = []
    dicts = []
    
    if(os.path.exists(folder)):
        print("loading")
        x_enc_in_tr = np.load(folder+'\\'+'x_enc_in_tr.npy')
        x_enc_in_te = np.load(folder+'\\'+'x_enc_in_te.npy')
        y_dec_in_tr = np.load(folder+'\\'+'y_enc_in_tr.npy')
        y_dec_in_te = np.load(folder+'\\'+'y_enc_in_te.npy')
        y_dec_out_tr = np.load(folder+'\\'+'y_dec_out_tr.npy')
        y_dec_out_te = np.load(folder+'\\'+'y_dec_out_te.npy')
        
        f_x_item_to_idx = open(folder+'\\'+'x_item_to_idx.pkl','rb')
        x_item_to_idx = pickle.load(f_x_item_to_idx)
        f_y_item_to_idx = open(folder+'\\'+'y_item_to_idx.pkl','rb')
        y_item_to_idx = pickle.load(f_y_item_to_idx)
        
        print("loaded")
        data = [[x_enc_in_tr, x_enc_in_te],
                [y_dec_in_tr, y_dec_in_te],
                [y_dec_out_tr, y_dec_out_te]]
        dicts = [x_item_to_idx, y_item_to_idx]      
    else:
        #read from file
        x_sents = read_sentences('.\..\Data\\'+x_filename+'.txt', portion)
        y_sents = read_sentences('.\..\Data\\'+y_filename+'.txt', portion)

        x_enc_in, y_dec_in, y_dec_out = None, None, None

        if in_level_x == 'w':
            x_sents = get_words(x_sents)
        elif in_level_x == 'w_ch':
            x_sents = get_words(x_sents)
        if in_level_y == 'w':
            y_sents = mark_start_end(y_sents, 'SOS ', ' EOS')
            y_sents = get_words(y_sents)
        else:
            y_sents = mark_start_end(y_sents, '\t', '\n')

        x_enc_in, x_item_to_idx = prepare_tensor(x_sents, is_w_ch=in_level_x == 'w_ch')
        y_dec_in, y_item_to_idx = prepare_tensor(y_sents, is_w_ch=in_level_y == 'w_ch')
        y_dec_out, _ = prepare_tensor(y_sents, 1, in_level_y == 'w_ch') 
        
        print("="*100)
        print(len(x_enc_in[0]), len(x_item_to_idx))
        print(x_enc_in.shape)
        print(y_dec_in.shape)
        print(x_item_to_idx, y_item_to_idx)
        
        x_enc_in_tr, x_enc_in_te , y_dec_in_tr, y_dec_in_te, y_dec_out_tr, y_dec_out_te = train_test_split(x_enc_in, 
                                                                   y_dec_in, 
                                                                   y_dec_out, 
                                                                   test_size = test_size)
        data = [[x_enc_in_tr, x_enc_in_te],
                [y_dec_in_tr, y_dec_in_te],
                [y_dec_out_tr, y_dec_out_te]]
        dicts = [x_item_to_idx, y_item_to_idx]
        
        #save files
        print("saving")
        os.makedirs(folder)
        save_np_array(folder +'\\x_enc_in_tr.npy',x_enc_in_tr)
        save_np_array(folder +'\\x_enc_in_te.npy',x_enc_in_te)
        save_np_array(folder +'\\y_enc_in_tr.npy',y_dec_in_tr)
        save_np_array(folder +'\\y_enc_in_te.npy',y_dec_in_te)
        save_np_array(folder +'\\y_dec_out_tr.npy',y_dec_out_tr)
        save_np_array(folder +'\\y_dec_out_te.npy',y_dec_out_te)
        
        f_x_item_to_idx= open(folder+'\\x_item_to_idx.pkl',"wb")
        f_y_item_to_idx= open(folder+'\\y_item_to_idx.pkl',"wb")
        pickle.dump(x_item_to_idx,f_x_item_to_idx)
        pickle.dump(y_item_to_idx,f_y_item_to_idx)
        f_x_item_to_idx.close()
        f_y_item_to_idx.close()
        print("saved")
        
    return data, dicts, folder


# In[5]:


# prepare_dataset('sentences_bad','sentences_good',0.0019, in_level_x = 'w_ch', in_level_y = 'w')


# In[ ]:





# In[ ]:




