import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def get_corpus(tokens):
    """ Combine the list of tokens into a string

    :return: a long string that combines all the tokens together
    """
    corpus = ' '.join(tokens)
    return corpus

def get_tfidf_vector(corpus, min_n_gram, max_n_gram):
    """ Convert the text into 2-D matrix with TfidfVectorizer by calculating the word frequency

    :param corpus: input text
    :param min_n_gram: the min number words in a sequence
    :param max_n_gram: the max number words in a sequence
    :return: 2D tfidf matrix
    """
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(min_n_gram, max_n_gram))
    X = vectorizer.fit_transform(corpus)

    features = vectorizer.get_feature_names()
    tfidf_matrix = X.toarray()
    return tfidf_matrix, features

def get_top_tfidf_features(tfidf_vec, features, top_n):
    """ Get top n tfidf values in row and return them with their corresponding feature names.

    :param tfidf_vec: 2D numpy array, tfidf vectors generated from get_tfidf_vector function
    :param features: a list of tfidf features
    :param top_n: top n tfidf features (with highest values)
    :return: a dataframe with top n features and its tfidf value
    """
    top_n_idxs = np.argsort(tfidf_vec)[::-1][:top_n]
    top_features = [(features[i], tfidf_vec[i]) for i in top_n_idxs]
    df = pd.DataFrame(top_features)
    df.columns = ['feature', 'tfidf_value']
    return df


def get_top_features_in_post(data, features, text_idx, top_n):
    """ Get top tfidf features in a specific post (matrix row)
    :param data: input dataframe
    :param features: a list of tfidf features
    :param text_idx: the index of the text
    :param top_n: top n tfidf features
    :return: a dataframe with top n features and its tfidf value
    """
    text = np.squeeze(data[text_idx])
    return get_top_tfidf_features(text, features, top_n)


def top_mean_features_by_label(X, y, features, top_n, label_id):
    """ Calculate the average of tfidf value of each feature among the posts and return a list of features
        with highest average.

    :param X: 2d numpy array (tfidf vectors)
    :param y: input labels ([0, 1, 0, 0, ...])
    :param features: a list of tfidf features
    :param top_n: top n tfidf features
    :param label_ids: the label of the specific group (stress (1) / non-stress (0))
    :return: a dataframe with top n features and their tfidf values
    """
    ids = np.where(y == label_id)

    if ids:
        new_data = X[ids]
    else:
        new_data = X

    tfidf_means = np.mean(new_data, axis=0)
    feature_df = get_top_tfidf_features(tfidf_means, features, top_n)
    feature_df['label'] = label_id

    return feature_df