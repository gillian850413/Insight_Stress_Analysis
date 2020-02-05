from nltk.corpus import stopwords
from nltk.corpus import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

class Post:
    def __init__(self, text):
        self.text = text
        return

    def tokenization(self):
        tokenizer = RegexpTokenizer(r'[a-zA-Z]{2,}')
        tokens = tokenizer.tokenize(self.text.lower())
        return tokens

    def remove_stopwords(self, tokens):
        words = [w for w in tokens if w not in stopwords.words('english')]
        return words

    def word_lemmatizer(self, tokens):
        lemmatizer = WordNetLemmatizer()
        lem_text = [lemmatizer.lemmatize(i) for i in tokens]
        return lem_text

    def preprocess(self):
        tokens = self.tokenization()
        tokens_remove_sw = self.remove_stopwords(tokens)
        final_tokens = self.word_lemmatizer(tokens_remove_sw)

        return final_tokens



