import re
import os
import pickle

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


class TextProcessor:
    def __init__(self, path=''):
        self.path = path

        self.stopwords = stopwords.words('english')
        self.ps = PorterStemmer()

        if os.path.isfile(self.path + 'stemmed.pkl'):
            with open(self.path + 'stemmed.pkl', 'rb') as file:
                self.stemmed = pickle.load(file)
        else:
            self.stemmed = dict()

    def __del__(self):
        with open(self.path + 'stemmed.pkl', 'wb') as file:
            pickle.dump(self.stemmed, file)

    def process(self, text: str, allow_stopwords: bool = False):
        ret = []
        for word in re.split('[^a-zA-Z]', str(text).lower()):
            if (word.isalpha() and word not in self.stopwords) or allow_stopwords:
                if word not in self.stemmed:
                    self.stemmed[word] = self.ps.stem(word)
                ret.append(self.stemmed[word])
        return ' '.join(ret)
