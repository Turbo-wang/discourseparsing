#!/usr/bin/env python
import numpy as np
import json
from gensim.models import Word2Vec
vocab = set()
pos_set = set()
w2i_dic = {}
i2w_dic = {}
p2i_dic = {}

train = "one_vs_others_standard/train_sec_02_20_EntRel_to_Expansion.json"
# dev = "dev.json"
# test = "test.json"
ndims=300
pos_ndims = 50

w2v_file = "GoogleNews-vectors-negative300.bin"
wv = Word2Vec.load_word2vec_format(w2v_file,binary=True)
def data_proc(f):
    fo=open(f)
    relation=[json.loads(x) for x in fo]
    fo.close()
    data=[]
    for r in relation:
        temp={}
        temp["Arg1"]=r["Arg1"]["Word"]
        temp["Arg2"]=r["Arg2"]["Word"]
        temp["Label"]=r["Label"]
        temp["POS1"]=r["Arg1"]["POS"]
        temp["POS2"]=r["Arg2"]["POS"]
        data.append(temp)
    fo.close()
    return data

def vocab_process(file):
    data=data_proc(file)
    for x in data:
        for w in x['Arg1']+x['Arg2']:
            vocab.add(w)
        for y in x['POS1']+x['POS2']:
            pos_set.add(y)


def WE_process():
    vocab_process(train)
    # vocab_process(test)
    # vocab_process(dev)
    idx = 1 # o for unknown
    for w in vocab:
        w2i_dic[w] = idx
        i2w_dic[idx] = w
        idx+=1
     #hyperparameter
    WE = np.zeros((len(vocab)+1,ndims),dtype='float32')
    pos_WE = np.zeros((len(pos_set)+1,pos_ndims),dtype='float32')
    pre_trained = set(wv.vocab.keys())
    for x in vocab:
        if x in pre_trained:
            WE[w2i_dic[x],:] = wv[x]
        else:
            WE[w2i_dic[x],:] = np.array(np.random.uniform(-0.5/ndims,0.5/ndims,(ndims,)),dtype='float32')   #hyperparameter
    for i,y in enumerate(pos_set):
        p2i_dic[y] = i
        pos_WE[i,:] = np.array(np.random.uniform(-0.5/pos_ndims,0.5/pos_ndims,(pos_ndims,)),dtype='float32')
    return WE,pos_WE

def embedding_process(file):
    data=data_proc(file)
    tmp = []
    for x in data:
        arg1 = []
        arg2 = []
        pos1 = []
        pos2 = []
        for w in x['Arg1']:
            arg1.append(w2i_dic[w])
        for w in x['Arg2']:
            arg2.append(w2i_dic[w])
        for w in x['POS1']:
            pos1.append(p2i_dic[w])
        for w in x['POS2']:
            pos2.append(p2i_dic[w])
        tmp.append((arg1,arg2,pos1,pos2,x["Label"]))
    data = tmp
    X_1 = np.array([x[0] for x in data])
    X_2 = np.array([x[1] for x in data])
    X_pos_1 = np.array([x[2] for x in data])
    X_pos_2 = np.array([x[3] for x in data])
    y = np.array([int(x[-1]) for x in data])
    # y = np_utils.to_categorical(np.array([x[2] for x in data]))
    X_1 = sequence.pad_sequences(X_1, maxlen=maxlen,padding='pre', truncating='pre')
    X_2 = sequence.pad_sequences(X_2, maxlen=maxlen,padding='post', truncating='post')
    X_pos_1 = sequence.pad_sequences(X_pos_1, maxlen=maxlen,padding='pre', truncating='pre')
    X_pos_2 = sequence.pad_sequences(X_pos_2, maxlen=maxlen,padding='post', truncating='post')
    print(file,(X_1.shape,X_pos_1.shape,y.shape))
    return(X_1,X_2,X_pos_1,X_pos_2,y)

#arg1,arg2,y,WE
# WE,pos_WE=WE_process()
# X_train_1,X_train_2,X_train_pos_1,X_train_pos_2,y_train=embedding_process(train)
# X_dev_1,X_dev_2,X_dev_pos_1,X_dev_pos_2,y_dev=embedding_process(dev)
# X_test_1,X_test_2,X_test_pos_1,X_test_pos_2,y_test=embedding_process(test)



# print (len(vocab))
# print('Build model...')