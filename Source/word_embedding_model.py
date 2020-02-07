import pandas as pd
import bentoml
from bentoml.artifact import PickleArtifact
from bentoml.handlers import DataframeHandler
from data_preprocess import Posts
from word_embedding_vectorizer import WordEmbeddingVectorizer
from gensim.models import Word2Vec

@bentoml.artifacts([PickleArtifact('word_vectorizer'),
                    PickleArtifact('word_embedding_rf')]) 

@bentoml.env(pip_dependencies=["pandas", "numpy", "gensim", "scikit-learn", "nltk"])

class WordEmbeddingModel(bentoml.BentoService):
        
    @bentoml.api(DataframeHandler, typ='series')
    def preprocess(self, series):
        preprocess_series = Posts(series).preprocess()
        input_matrix = self.artifacts.word_vectorizer.fit(preprocess_series).transform(preprocess_series)
        return input_matrix
    
    @bentoml.api(DataframeHandler, typ='series')
    def predict(self, series):
        input_matrix = self.preprocess(series)
        pred_labels = self.artifacts.word_embedding_rf.predict(input_matrix)
        pred_proba = self.artifacts.word_embedding_rf.predict_proba(input_matrix)
        confidence_score = [prob[1] for prob in pred_proba]
        output = pd.DataFrame({'text': series, 'confidence_score': confidence_score, 'labels': pred_labels})
        output['labels'] = output['labels'].map({1: 'stress', 0: 'non-stress'})
        
        return output
