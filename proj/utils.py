import re
import os
import pickle

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


class TextProcessor:
    """
    Class for carrying all the text pre-processing stuff throughout the project
    """

    def __init__(self, path: str = ''):
        """
        :param path: path to stemmed.pkl file -
        serialized with pickle dict with mappings from raw word to stemmed version of it.
        Used to sped things up because NLTK's PorterStemmer is relatively slow
        """
        self.path = path

        self.stopwords = stopwords.words('english')
        self.ps = PorterStemmer()

        # load stemmed words if existent
        if os.path.isfile(self.path + 'stemmed.pkl'):
            with open(self.path + 'stemmed.pkl', 'rb') as file:
                self.stemmed = pickle.load(file)
        else:
            # otherwise, stemmer will be used for each unique word again
            self.stemmed = dict()

    def __del__(self):
        # save the updated mappings
        with open(self.path + 'stemmed.pkl', 'wb') as file:
            pickle.dump(self.stemmed, file)

    def process(self, text: str, allow_stopwords: bool = False) -> str:
        """
        Process the specified text,
        splitting by non-alphabetic symbols, casting to lower case,
        removing stopwords and stemming each word

        :param text: text to precess
        :param allow_stopwords: whether to remove stopwords
        :return: processed text
        """
        ret = []

        # split and cast to lower case
        for word in re.split('[^a-zA-Z]', str(text).lower()):
            # remove non-alphabetic and stop words
            if (word.isalpha() and word not in self.stopwords) or allow_stopwords:
                if word not in self.stemmed:
                    self.stemmed[word] = self.ps.stem(word)
                # use stemmed version of word
                ret.append(self.stemmed[word])
        return ' '.join(ret)
