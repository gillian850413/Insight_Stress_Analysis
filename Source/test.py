from preprocess import Posts
import tfidf
import pandas as pd

path = '/Users/gillianchiang/Desktop/Insight/Project/Insight_Stress_Analysis/data/'
train = pd.read_csv(path + 'dreaddit-train.csv', encoding = "ISO-8859-1")


train_text = train['text']
text = Posts(train_text)

processed_text = text.preprocess()


corpus = processed_text.apply(lambda x: tfidf.get_corpus(x))
X_train, features = tfidf.get_tfidf_vector(corpus, 1, 1)

print(X_train.shape)


non_stress_df = tfidf.top_mean_features_by_label(X_train, train['label'], features, 200, 0)
print(non_stress_df)