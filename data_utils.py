import json, os
import gensim
from collection import Counter

class DataUtil:

    def get_sentence_pairs(self, filename, filepath):
        self.json_list = []
        with open(os.path.join(filepath, filename)) as f:
            for line in f:
                jsonobj = json.loads(line.strip())
            # print json.dumps(jsonobj, indent=2)
            # break
                self.json_list.append(jsonobj)

    def load_word2vec(self, file_name):
        self.model = gensim.models.Word2Vec.load_word2vec_format(file_name, binary=True)

    def generate_xxy(self, json_list):
        X_indexs = []
        X_pos = []

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
    d.get_sentence_pairs("dev_sec_00_01.json", "one_vs_others_standard")
    d.ana_pos_list()