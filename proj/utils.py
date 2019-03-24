import re

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


class TextProcessor:
    def __init__(self):
        self.stopwords = stopwords.words('english')
        self.ps = PorterStemmer()
        self.stemmed = dict()

    def process(self, text: str):
        ret = []
        for word in re.split('[^a-zA-Z]', str(text).lower()):
            if word.isalpha() and word not in self.stopwords:
                if word not in self.stemmed:
                    self.stemmed[word] = self.ps.stem(word)
                ret.append(self.stemmed[word])
        return ' '.join(ret) + ' '
