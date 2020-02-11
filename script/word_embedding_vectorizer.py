"""
Create a tfidf word embedding vectorizer for Word2Vec model
Reference: https://towardsdatascience.com/nlp-performance-of-different-word-embeddings-on-text-classification-de648c6262b
"""

import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


class WordEmbeddingVectorizer(object):

    def __init__(self, word_model):

        self.word_model = word_model
        self.word_idf_weight = None
        self.vector_size = word_model.wv.vector_size

    def fit(self, texts):
        """ Build a word embedding model with Word2Vec which uses tfidf as weight

        :param texts: a list of preprocessed texts
        :return: the object
        """
        text_docs = []
        for text in texts:
            text_docs.append(" ".join(text))

        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,1)) # default 1-gram
        tfidf.fit(text_docs)  # must be list of text string

        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of known idf's
        max_idf = max(tfidf.idf_)  # used as defaultdict's default value
        self.word_idf_weight = defaultdict(lambda: max_idf,
                                           [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()])
        return self


    def transform(self, texts):
        """ Transform the text, follows sklearn transform

        :param texts:  a list of preprocessed texts
        :return: a word vector
        """
        text_word_vector = self.word_average_list(texts)
        return text_word_vector

    def word_average(self, sent):
        """ Compute average word vector for a single doc/sentence. (use for doc2vec)
        :param sent: list of sentence tokens
        :return: the mean of averaging word vectors
        """
        mean = []
        for word in sent:
            if word in self.word_model.wv.vocab:
                mean.append(self.word_model.wv.get_vector(word) * self.word_idf_weight[word])  # idf weighted

        if not mean:  # empty words
            # If a text is empty, return a vector of zeros.
            logging.warning("cannot compute average owing to no vector for {}".format(sent))
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean


    def word_average_list(self, docs):
        """ Compute average word vector for multiple docs, where docs had been tokenized. (use for doc2vec)
        :param docs: list of sentence in list of separated tokens
        :return: an array of average word vector in shape (len(docs),)
        """
        return np.vstack([self.word_average(sent) for sent in docs])