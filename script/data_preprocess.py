from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.corpus import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

# setup with config.py
# import nltk

# nltk.download("stopwords")
# nltk.download("wordnet")

class Posts:
    def __init__(self, text_df):
        self.text_df = text_df
        return

    def tokenization(self):
        """ Tokenize the contents of the posts and remove the strings that contains punctuations,
            numbers or only single letter

        :return: a dataframe which each row becomes list of tokens
        """
        
        tqdm.pandas()
        tokenizer = RegexpTokenizer(r'[a-zA-Z]{2,}')
        tokens_df = self.text_df.progress_apply(lambda x: tokenizer.tokenize(x.lower()))
        return tokens_df

    def remove_stopwords(self, tokens):
        """ Remove the stopwords from the list of tokens

        :param tokens: a list of tokens
        :return: a list of tokens without stopwords
        """
        words = [w for w in tokens if w not in stopwords.words('english')]
        return words

    def word_lemmatizer(self, tokens):
        """ Conduct lemmatization on the list of tokens so that different inflected
            forms of a word can be analysed as a single item.

        :param tokens: a list of tokens
        :return:
        """
        lemmatizer = WordNetLemmatizer()
        lem_text = [lemmatizer.lemmatize(i) for i in tokens]
        return lem_text

    def preprocess(self):
        """ Combine all the preprocess steps into one function.

        :return: a dataframe that each row is a list of tokens that have removed punctuations and stopwords,
                 and have done lemmatization
        """
        tokens = self.tokenization()
        tokens_sw = tokens.progress_apply(lambda x: self.remove_stopwords(x))
        final_tokens = tokens_sw.progress_apply(lambda x: self.word_lemmatizer(x))
        return final_tokens



