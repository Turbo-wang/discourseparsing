#!/env/bin python
# -*- coding=utf8 -*-

import numpy as np
import os, sys, re
import nltk
import pickle, codecs
import jpype
from gensim import models
from PSDPredictFuncs import extfea_trn, extfea_tst, eval, update_tst_feature, print_for_check, explain_feature, extfea_trn_pun
from sklearn import linear_model
import tensorflow as tf
import math, collections
import data_utils, config
import SimplyMLP
import pyltp
# feature_select = [ "0floor", "0ceil", "3floor", "3ceil", "4floor", "4ceil" , "prewlen", "curwlen", "nxtwlen"]

def load_word2vec(filename):
    model = models.Word2Vec.load_word2vec_format(filename)
    return model

def convert_feature_to_vector(train_fealist, model, feature_select):
    pinyin_one_hot_dict = build_pinyin_index_dict()
    pos_one_hot_dict = build_one_hot_pos()
    X = []
    Y = []
    for fealist in train_fealist:
        fea_vector = []
        res = fealist[1]
        fealist_od = fealist[0]
        for fea_name in feature_select:
            if fea_name.endswith("word"):
                tmp = model[fealist_od[fea_name]] if fealist_od[fea_name] in model else [0] * model.vector_size
                fea_vector = np.concatenate((fea_vector, tmp))
            elif fea_name.endswith("pos"):
                tmp = pos_one_hot_dict[fealist_od[fea_name]] if fealist_od[fea_name] in pos_one_hot_dict else [0] * len(pos_one_hot_dict['v'])
                fea_vector = np.concatenate((fea_vector, tmp))
            else:
                fea_vector = np.concatenate((fea_vector, np.array([fealist_od[fea_name]])))
        X.append(fea_vector)
        Y.append(res)
    return X, Y


def convert_feature_to_index_vector(train_fealist, word2index, feature_select):
    pinyin_one_hot_dict = build_pinyin_index_dict()
    pos_one_hot_dict = build_one_hot_pos()
    X_index = []
    X_others = []
    Y = []
    for fealist in train_fealist[:-1]:
        index_vector = []
        others_vector = []
        word_list = []
        pos_list = []
        res = fealist[1]
        fealist_od = fealist[0]
        for fea_name in feature_select:
            # if fea_name in feature_select:
            if fea_name.endswith("word"):
                word = fealist_od[fea_name]
                word_list.append(word)
            elif fea_name.endswith("pos"):
                pos = fealist_od[fea_name]
                pos_list.append(pos)
            else:
                # others_vector = np.concatenate(others_vector, np.array([fealist_od[fea_name]]))
                pass
        for index, word in enumerate(word_list):
            if index == 0 and word == '':
                index_vector.append(word2index["SENTENCE_START"])
            elif index == len(word_list) - 1 and word == '':
                index_vector.append(word2index["SENTENCE_END"])
            else:
                index_vector.append(word2index[word]) if word in word2index else index_vector.append(word2index["UNKNOWN_TOKEN"])
        for pos in pos_list:
            others_vector.append(pos_one_hot_dict[pos]) if pos in pos_one_hot_dict else others_vector.append(np.array([0] * len(pos_one_hot_dict['v'])))
        X_index.append(index_vector)
        X_others.append(others_vector)
        if int(res) == 0:
            Y.append([0, 1])
        else:
            Y.append([1, 0])
    return X_index, X_others, Y


def test_feature_to_vector(fealist_od, model, pinyin_one_hot_dict, pos_one_hot_dict):
    fea_vector = []
    # feature_select = ["preword", "curword", "nxtword", "prepos", "curpos", "nxtpos", "0floor", "0ceil", "1floor", "1ceil", "3floor", "3ceil", "4floor", "4ceil", "prewlen", "curwlen", "nxtwlen"]
    for fea_name in feature_select:
        if fea_name.endswith("word"):
            tmp = model[fealist_od[fea_name]] if fealist_od[fea_name] in model else [0] * model.vector_size
            fea_vector = np.concatenate((fea_vector, tmp))
        elif fea_name.endswith("pos"):
            tmp = pos_one_hot_dict[fealist_od[fea_name]] if fealist_od[fea_name] in pos_one_hot_dict else [0] * len(pos_one_hot_dict['v'])
            fea_vector = np.concatenate((fea_vector, tmp))
        else:
            fea_vector = np.concatenate((fea_vector, np.array([fealist_od[fea_name]])))
    return fea_vector


def build_pinyin_index_dict():
    vowel_to_index = {}
    consonant_to_index = {}
    consonant_list = ['b','p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w']
    vowel_list = ['ang', 'ei', 've', 'ai', 'in', 'iu', 'ong', 'ao', 'an', 'uai', 'en', 'iong', 'uan', 'ia', 'ing', 'ie', 'ian', 'eng', 'iang', 'ui', 'uang', 'a', 'iao', 'e', 'i', 'o', 'uo', 'un', 'u', 'v', 'ue', 'ou', 'ua']
    for index, con in enumerate(consonant_list):
        consonant_to_index[con] = index
    for index, con in enumerate(vowel_list):
        vowel_to_index[con] = index
    return consonant_to_index, vowel_to_index


def build_one_hot_pinyin(consonant_to_index, vowel_to_index):
    pinyin_one_hot_dict = {}
    for key, value in consonant_to_index.items():
        vector = np.zeros(len(consonant_to_index))
        vector[value] = 1
        pinyin_one_hot_dict[key] = vector
    for key, value in vowel_to_index.items():
        vector = np.zeros(len(vowel_to_index))
        vector[value] = 1
        pinyin_one_hot_dict[key] = vector
    return pinyin_one_hot_dict


def build_one_hot_pos():
    pos_list = ['Ag', 'a', 'ad', 'an', 'Bg', 'b', 'c', 'Dg', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'Mg', 'm', 'Ng', 'n', 'nr', 'ns',
    'nt', 'nx', 'nz', 'o', 'p', 'Qg', 'q', 'Rg', 'r', 's', 'Tg', 't', 'Ug', 'u', 'Vg', 'v', 'vd', 'vn', 'w', 'x', 'Yg', 'y', 'z']
    pos_one_hot_dict = {}
    pos_to_index = {}
    for index, pos in enumerate(pos_list):
        pos_to_index[pos] = index
    for key, value in pos_to_index.items():
        vector = np.zeros(len(pos_to_index))
        vector[value] = 1
        pos_one_hot_dict[key] = vector
    return pos_one_hot_dict


def construct_batch_matrix(_X_indexs, _X_others, _Y, xid, batch_size):
    X_indexs = []
    X_others = []
    Y = []
    for i in xid:
        X_indexs.append(_X_indexs[i])
        X_others.append(_X_others[i])
        Y.append(_Y[i])
    assert len(X_indexs) == len(Y) and len(X_others) == len(X_indexs)
    X_indexs_batch_list = []
    X_others_batch_list = []
    Y_batch_list = []
    total_batch = int(len(Y)/batch_size)
    for i in range(int(total_batch)):
        X_indexs_batch_list.append(X_indexs[i * batch_size: (i+1) * batch_size])
        X_others_batch_list.append(X_others[i * batch_size: (i+1) * batch_size])
        Y_batch_list.append(Y[i * batch_size: (i+1) * batch_size])
    X_indexs_batch_list.append(X_indexs[(i+1)*batch_size:])
    X_others_batch_list.append(X_others[(i+1)*batch_size:])
    Y_batch_list.append(Y[(i+1)*batch_size:])
    return X_indexs_batch_list, X_others_batch_list, Y_batch_list


def build_word2index_index2word(min_count=1):
    model = load_word2vec("Corpus/word2vec_model/only_curr_50dim.model")
    seg = pyltp.Segmentor()
    seg.load("Corpus/ltp/ltp_data/cws.model")
    file_list = ["ProsodyTrain.txt", "ProsodyTest.txt"]
    word2index = {}
    index2word = []
    word_list_file = []
    i = 0
    for file in file_list:
        with codecs.open(os.path.join("Corpus/King-TTS_labelAll", file), 'r', 'utf8') as f:
            for line_num, line_con in enumerate(f):
                if (line_num + 3) % 3 == 0:
                    con = re.sub('#\d', "", line_con.strip().split()[1]).encode("utf8")
                    word_list = seg.segment(con)
                    word_list_file += word_list
    word_frq = collections.Counter(word_list_file)

    for word in word_list_file:
        if word.decode("utf8") in model and word not in word2index and word_frq[word] >= min_count:
            index2word.append(word)
            word2index[word] = i
            i += 1
    index2word.append("UNKNOW_TOKEN")
    word2index["UNKNOWN_TOKEN"] = i
    i += 1
    index2word.append("SENTENCE_START")
    word2index["SENTENCE_START"] = i
    i += 1
    index2word.append("SENTENCE_END")
    word2index["SENTENCE_END"] = i
    return word2index, index2word


def build_embedding_matrix(word2index, index2word):
    model = load_word2vec("Corpus/word2vec_model/only_curr_50dim.model")

    embedding_matrix = []
    for word in index2word[:-3]:
        word = word.decode("utf8")
        embedding_matrix.append(model[word])
    embedding_matrix.append([0]*model.vector_size) #UNKNOWN_TOKEN
    embedding_matrix.append([0]*model.vector_size) #SENTENCE_START
    embedding_matrix.append([1]*model.vector_size) #SENTENCE_END
    embedding_matrix = np.asarray(embedding_matrix, dtype="float32")
    return embedding_matrix




def load_fealist(train_or_dev):
    train_data = {}
    jclassins = None
    model = load_word2vec("Corpus/word2vec_model/only_curr_50dim.model")
    clfroot = 'Result/ProsodyPredictionClassifier'
    fearoot = 'Result/ProsodyPredictionFeature'
    if train_or_dev:
        print 'load Training data...'
        poslines, psdlines, pinlines = data_utils.loadCorups(config.prosodytrainfilesm)
    else:
        print 'load Dev data...'
        poslines, psdlines, pinlines = data_utils.loadCorups(config.prosodytestfilesm)
    if not os.path.exists(clfroot):
        os.mkdir(clfroot)
    # fealist = extfea_trn(psdlines, pinlines, fearoot, jclassins)
    fealist = extfea_trn_pun(psdlines, fearoot)
    fealist_dict = {}
    # for level in [1]:
    for level in range(1, 5):
        trnfealist = []
        pos, neg = 0, 0
        for fea in fealist:
            if fea[3] == level - 1:
                neg += 1
                trnfealist.append((fea[2], 0))
            if fea[3] >= level:
                pos += 1
                trnfealist.append((fea[2], 1))
        fealist_dict[level] = trnfealist
        print 'Prosody level = %s pos:%d neg:%d' % (str(level), pos, neg)
    return fealist_dict


def train_LR_models(feature_select):
    train_data = {}
    jclassins = None
    model = load_word2vec("Corpus/word2vec_model/only_curr_50dim.model")
    print 'Training ...'
    clfroot = 'Result/ProsodyPredictionClassifier'
    fearoot = 'Result/ProsodyPredictionFeature'
    poslines, psdlines, pinlines = data_utils.loadCorups(config.prosodytrainfile)
    if not os.path.exists(clfroot):
        os.mkdir(clfroot)
    fealist = extfea_trn_pun(psdlines, fearoot)
#     for level in [4]:
    for level in range(1, 2):
        trnfealist = []
        pos, neg = 0, 0
        for fea in fealist:
            if fea[3] == level - 1:
                neg += 1
                trnfealist.append((fea[2], 0))
            if fea[3] >= level:
                pos += 1
                trnfealist.append((fea[2], 1))
        print 'Prosody level = %s pos:%d neg:%d' % (str(level), pos, neg)
        X, Y = convert_feature_to_vector(trnfealist, model, feature_select)
        # train_data["X%s" % level] = X
        # train_data["Y%s" % level] = Y
        # # construct_batch_matrix(X, Y, 256)
    # return train_data
        logreg = linear_model.LogisticRegression(C=1e2)
        logreg.fit(X, Y)
        pickle.dump(logreg, open(os.path.join(clfroot, 'logreg_level_' + str(level) + '.pickle'), 'wb'))


def test_LR_models():
    pinyin_one_hot_dict = build_pinyin_index_dict()
    pos_one_hot_dict = build_one_hot_pos()
    model = load_word2vec("Corpus/word2vec_model/only_curr_50dim.model")
    tstfiles = 'Corpus/TstFiles.txt'
    print '\nTesting ...'#, itr
    jclassins = None
    refroot = 'Corpus/King-TTS-025_King_label/ReferenceProsodyPredictionFeature'
    # posroot = 'Result/ICTCLAS_POSMatched'
    psdroot = 'Corpus/King-TTS-025_King_label/ProsodyLabeling'
    posroot = 'Corpus/King-TTS-025_King_label/POSLabeling'
    fearoot = 'Result/ProsodyPredictionFeature'
    clfroot = 'Result/ProsodyPredictionClassifier'
    resroot = 'Result/ProsodyPredictionResult'

    if not os.path.exists(fearoot):
        os.mkdir(fearoot)
    if not os.path.exists(resroot):
        os.mkdir(resroot)
    recall_dict = {}
    precision_dict = {}
    fmeasure_dict = {}
    tstfeadict = dict()
    for filename in open(tstfiles, 'r').readlines():
        filename = filename.strip()
        tstfeadict[filename] = extfea_tst(lines, poslines, jclassins)

    # 更新测试文件特征
    update_tst_feature(tstfeadict, fearoot, resroot)

    for level in range(1, 5):
        print 'Prosody level = ' + str(level)

        logreg = pickle.load(open(os.path.join(clfroot, 'logreg_level_' + str(level) + '.pickle'), 'rb'))
        # classifier = pickle.load(open(os.path.join(clfroot, 'classifier_level_' + str(level) + '.pickle'), 'rb'))
        # classifier = pycrfsuite.Tagger()
        # classifier.open(os.path.join(clfroot, 'classifier_level_' + str(level) + '.crfsuite'))

        for filename in open(tstfiles, 'r').readlines():
            resfilepath = os.path.join(resroot, filename.strip()).replace('.txt', '_')
            outfile = open(resfilepath + str(level) + '.txt', 'w+')
            prereslines = (open(resfilepath + str(level - 1) + '.txt', 'r').readlines() if level > 1 else [])

            fealist = tstfeadict[filename.strip()]
            for i in range(len(fealist)):
                fea = fealist[i]
                outfile.write(str(fea[0]) + '\t' + str(fea[1]) + '\t')

                if level > 1 and prereslines[i].strip()[-1] == '0':
                    outfile.write('0\n')
                    continue

                # pdist = classifier.prob_classify(fea[2])
                vector = test_feature_to_vector(fea[2], model, pinyin_one_hot_dict, pos_one_hot_dict)
                pdist = logreg.predict(np.reshape(vector, (1,-1)))
                # tstres = (0 if pdist.prob(0) > pdist.prob(1) else 1)
                tstres = pdist[0]
                outfile.write(str(tstres) + '\n')

                # tstres = classifier.tag([fea[2]])
                # outfile.write(tstres[0] + '\n')

            outfile.close()

        (recall, precision, fmeasure) = eval(tstfiles, resroot, refroot, level)
        recall_dict[level] = recall
        precision_dict[level] = precision
        fmeasure_dict[level] = fmeasure

        print '\trecall = ' + str(recall)
        print '\tprecision = ' + str(precision)
        print '\tf-measure = ' + str(fmeasure)
# 输出格式化停顿信息
    print_for_check(tstfeadict, resroot)
    return (recall_dict, precision_dict, fmeasure_dict)




def valudation_feature():
    feature_select = ["preword", "curword", "nxtword", "prepos", "curpos", "nxtpos",  "0floor", "0ceil", "3floor", "3ceil", "4floor", "4ceil", "prewlen", "curwlen", "nxtwlen"]
    # feature_select = ["prepos", "curpos"]
    feature_use = []
    recall_trend = {}
    precision_trend = {}
    f_trend = {}
    for level in range(1, 5):
        recall_trend[level] = []
        precision_trend[level] = []
        f_trend[level] = []
    for j in range(len(feature_select)):
        for i in range(len(feature_select)):
            if i != j:
                feature_use.append(feature_select[i])
        load_train_data(feature_select)
        (recall, precision, fmeasure) = test_LR_models()
        for level in range(1, 5):
            recall_trend[level].append(recall[level])
            precision_trend[level].append(precision[level])
            f_trend[level].append(fmeasure[level])
    f = open("trenday.out", 'w')
    for i in range(1, 5):
        f.write(str(i))
        f.write("\n")
        for j in range(len(feature_select)):
            f.write("\t")
            f.write(feature_select[j])
        f.write("\n")
        recall_tr = recall_trend[i]
        f.write("recall")
        for recall in recall_tr:
            f.write("\t")
            f.write(str(recall))
        f.write("\n")
        f.write("precision")
        precision_tr = precision_trend[i]
        for precision in precision_tr:
            f.write("\t")
            f.write(str(precision))
        f.write("\n")
        f.write("fmeasure")
        f_tr = f_trend[i]
        for f_ in f_tr:
            f.write("\t")
            f.write(str(f_))
        f.write("\n")
    f.close()

if __name__ == '__main__':
    feature_select = ["pprepos", "prepos", "curpos", "nxtpos", "nnxtpos"]
    # train_MLP_model()
    word2index, index2word =  build_word2index_index2word()
    train_LR_models(feature_select)
    #test_MLP_model()
    # # load_train_data(feature_select)
    (recall, precision, fmeasure) = test_LR_models()
