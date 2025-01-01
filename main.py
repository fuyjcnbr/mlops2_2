import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# import seaborn as sns
# import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn


if not nltk.data.find('corpora/stopwords.zip'):
    nltk.download("stopwords")

if not nltk.data.find('corpora/wordnet.zip'):
    nltk.download("wordnet")


STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

EMBEDDINGS_DIM = 1
HASHING_FEATURES = 16 # должно быть >= EMBEDDINGS_DIM

N = 2000


def normilize_string(s):
    s = s.lower()
    s = s.translate(s.maketrans("", "", string.punctuation))
    li = [x for x in s.split() if x not in STOP_WORDS]
    li2 = [LEMMATIZER.lemmatize(x) for x in li]
    s = " ".join(li2)
    return s


def naive_embeddings(X):
  vectorizer = HashingVectorizer(n_features=HASHING_FEATURES)
  x_tmp = vectorizer.fit_transform(X)
  # print(x_tmp.shape)
  svd = TruncatedSVD(n_components=EMBEDDINGS_DIM, n_iter=7, random_state=33)
  x_tmp2 = svd.fit_transform(x_tmp)
  del x_tmp
  return x_tmp2


mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("RidgeClassifier 1")
mlflow.set_experiment("RandomForestClassifier 1")


with mlflow.start_run(run_name="RandomForestClassifier run 2"):
    random_state = 33
    np.random.seed(random_state)

    print(f"reading data...")
    data = pd.read_csv("train.csv")

    print(f"preparing data...")
    df = data[data[["Text", "Sentiment"]].notnull().all(1)]
    df = df[:N]
    X = df["Text"].apply(normilize_string)
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(df["Sentiment"]))

    print(f"train/test split...")
    X1_embeddings = naive_embeddings(X)
    X_train, X_test, y_train, y_test = train_test_split(X1_embeddings, y, test_size=0.3, random_state=random_state)


    print(f"training model...")
    # alpha = 5.5
    # alpha = 4.5
    # estimator = RidgeClassifier(alpha=alpha, solver="sparse_cg")
    # estimator.fit(X_train, y_train)

    n_estimators = 5
    estimator = RandomForestClassifier(n_estimators=n_estimators)
    estimator.fit(X_train, y_train)

    print(f"predicting...")
    y1_train_pred = estimator.predict(X_train)
    y1_test_pred = estimator.predict(X_test)

    print(f"counting metrics...")
    accuracy_train = accuracy_score(y_train, y1_train_pred)
    accuracy_test = accuracy_score(y_test, y1_test_pred)

    # mlflow.log_param("alpha", alpha)
    mlflow.log_param("n_estimators", n_estimators)

    mlflow.log_metric("accuracy_train", accuracy_train)
    mlflow.log_metric("accuracy_test", accuracy_test)

    mlflow.sklearn.log_model(estimator, "RidgeClassifier")

mlflow.end_run()