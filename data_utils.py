import json, os
import gensim
from collections import Counter

class DataUtil:

    def get_sentence_pairs(self, filename, filepath):
        self.json_list = []
        self.sentence_list1 = []
        self.sentence_list2 = []
        self.pos_list1 = []
        self.pos_list2 = []
        self.label_list = []
        with open(os.path.join(filepath, filename)) as f:
            for line in f:
                json_obj = json.loads(line.strip())
                arg1 = json_obj["Arg1"]
                arg2 = json_obj["Arg2"]
                pos_list1 = arg1["POS"]
                pos_list2 = arg2["POS"]
                sen_list1 = arg1["Word"]
                sen_list2 = arg2["Word"]
                connect = json_obj["Label"]
                self.sentence_list1.append(sen_list1)
                self.sentence_list2.append(sen_list2)
                self.pos_list1.append(pos_list1)
                self.pos_list2.append(pos_list2)
                self.label_list.append(connect)
            # print json.dumps(jsonobj, indent=2)
            # break
                # self.json_list.append(jsonobj)

    def get_train_data(self):
        self.load_word2vec("google_vectors.bin")
        self.get_sentence_pairs()
        self.batch_size = 128
        X1_inputs_list = []
        X2_inputs_list = []
        Y_inputs_list = []
        for sen1, sen2, label in zip(self.sentence_list1, self.sentence_list2, self.label_list):
            x1_sen = []
            flag = False
            for idx, word in enumerate(sen1):
                if idx==50:
                    break
                    flag = True
                if word in self.model:
                    x1_sen.append(self.model[word])
                else:
                    x1_sen.append(np.zeros(300), dtype="float32")
            if not flag:
                for i in range(len(x1_sen), 50):
                    x1_sen.append(np.zeros(300), dtype="float32")
            X1_inputs_list.append(x1_sen)
            x2_sen = []
            flag = False
            for idx, word in enumerate(sen2):
                if idx==50:
                    break
                    flag = True
                if word in self.model:
                    x2_sen.append(self.model[word])
                else:
                    x2_sen.append(np.zeros(300), dtype="float32")
            if not flag:
                for i in range(len(x2_sen), 50):
                    x2_sen.append(np.zeros(300), dtype="float32")
            X2_inputs_list.append(x2_sen)
            if labe == "1":
                Y_inputs_list.append(np.array([1, 0], dtype='float32'))
            else:
                Y_inputs_list.append(np.array([0, 1], dtype='float32'))

        return X1_inputs_list, X2_inputs_list, Y_inputs_list

    def construct_batch_matrix(self, X1_indexs, X2_indexs, Y, batch_size):
        # X1_indexs = []
        # X1_indexs = []
        # Y = []
        # for i in xid:
        #     X_indexs.append(_X_indexs[i])
        #     X_others.append(_X_others[i])
        #     Y.append(_Y[i])
        # assert len(X_indexs) == len(Y) and len(X_others) == len(X_indexs)
        X1_batch_list = []
        X2_batch_list = []
        Y_batch_list = []
        total_batch = int(len(_Y)/batch_size)
        for i in range(int(total_batch)):
            X1_batch_list.append(X1_indexs[i * batch_size: (i+1) * batch_size])
            X2_batch_list.append(X2_indexs[i * batch_size: (i+1) * batch_size])
            Y_batch_list.append(Y[i * batch_size: (i+1) * batch_size])
        X1_batch_list.append(X1_indexs[(i+1)*batch_size:])
        X2_batch_list.append(X2_indexs[(i+1)*batch_size:])
        Y_batch_list.append(Y[(i+1)*batch_size:])
        return X1_batch_list, X2_batch_list, Y_batch_list



    def load_word2vec(self, file_name):
        self.model = gensim.models.Word2Vec.load_word2vec_format(file_name, binary=True)

    # def generate_xxy(self, json_list):
    #     X_indexs = []
    #     X_pos = []

    def ana_pos_list(self):
        pos_list = []
        for json_obj in self.json_list:
            arg1 = json_obj["Arg1"]
            arg2 = json_obj["Arg2"]
            pos_list1 = arg1["POS"]
            pos_list2 = arg2["POS"]
            pos_list = pos_list1 + pos_list2
        print Counter(pos_list)


if __name__ == '__main__':
    d = DataUtil()
    d.get_sentence_pairs("dev_sec_00_01.json", "one_vs_others_standard/Comparison_vs_others")
    d.ana_pos_list()