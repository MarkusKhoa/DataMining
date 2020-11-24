#Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import re
import nltk
import string
import matplotlib.pyplot as plt
import pickle

from textblob import TextBlob
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from xgboost import XGBClassifier
from lightgbm import LGBMModel,LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df_test = pd.read_csv('./Dataset/drugsComTest_raw.csv')
df_train = pd.read_csv('./Dataset/drugsComTrain_raw.csv')

df_train['sentiment'] = df_train['rating'].apply(lambda ra: "negative" if ra <= 5 else "positive")
df_test['sentiment'] = df_test['rating'].apply(lambda ra: "negative" if ra <= 5 else "positive")

data = pd.concat([df_train, df_test])

#Xử lý stopwords
stops = set(stopwords.words('english'))
not_stops = ["aren't","couldn't","didn't","doesn't","don't","hadn't","haven't","isn't","mightn't","mustn't","needn't","no","nor","not","shan't","shouldn't","wasn't","weren't","won't","wouldn't"]
for not_stop in not_stops:
  stops.remove(not_stop)

stemmer = SnowballStemmer('english')
wnl = WordNetLemmatizer()


print('xu li stopword xong')
def clean_text(text):
  punct_removed = re.sub('[^a-zA-Z0-9\']', ' ', text)
  ascii_removed = re.sub('[^\x00-\x7F]+', ' ', punct_removed)
  words = ascii_removed.lower().split()
  needed_words = [word for word in words if word not in stops]
  last_text = [wnl.lemmatize(rw) for rw in needed_words]

  return " ".join(last_text)

def sentiment(review):
    # Sentiment polarity of the reviews
    pol = []
    for i in review:
        analysis = TextBlob(i)
        pol.append(analysis.sentiment.polarity)
    return pol


print('bat dau xu li data....')
data['cleaned_review'] = data['review'].apply(clean_text)

#Sinh các features mới biến đổi từ review và clean review
data['sentiment_polarity'] = sentiment(data['review'])
data['sentiment_clean'] = sentiment(data['cleaned_review'])

data['count_word']=data["cleaned_review"].apply(lambda x: len(str(x).split()))

data['count_unique_word']=data["cleaned_review"].apply(lambda x: len(set(str(x).split())))
data['count_letters']=data["cleaned_review"].apply(lambda x: len(str(x)))

data["count_punctuations"] = data["review"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
data["count_words_upper"] = data["review"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
data["count_words_title"] = data["review"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
data["count_stopwords"] = data["review"].apply(lambda x: len([w for w in str(x).lower().split() if w in stops]))

data["mean_word_len"] = data["cleaned_review"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


print('xu li data xong')

X_data = data[['usefulcount', 'sentiment_polarity', 'sentiment_clean', 'count_word', 'count_unique_word', 'count_letters',
                'count_punctuations', 'count_words_upper', 'count_words_title',
                'count_stopwords', 'mean_word_len']]

feature_inputs = X_data.columns
label = data['encoded_sentiment']
X_train, X_test, y_train, y_test = train_test_split(X_data, label, test_size = 0.3, random_state = 42)

clf = LGBMClassifier(n_estimators=10000,
        learning_rate=0.10, num_leaves=30, subsample=.9, max_depth=7, reg_alpha=.1,
        reg_lambda=.1, min_split_gain=.01,
        min_child_weight=2,
        silent=-1,
        verbose=-1,
        )


print('bat dau train')
LGBM_model = clf.fit(X_train, y_train)
filename = 'lgbm.pkl'
with open(filename, 'wb') as file:
  pickle.dump(LGBMModel, file)

print('Da luu model thanh cong')