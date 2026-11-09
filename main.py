# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import nltk
import re
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

from sklearn.model_selection import train_test_split
from lime.lime_text import LimeTextExplainer
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
stopwords = set(nltk.corpus.stopwords.words("english"))

#Creating a dataframe by loading the dataset
import os
DATA_DIR = os.environ.get("FND_DATA_DIR", "Prototype/data/raw")
df_Fake = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
df_True = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))

# Add labels before merging
df_Fake["label"] = 0     # Fake = 0
df_True["label"] = 1     # True = 1

# Merge both
df = pd.concat([df_Fake, df_True], axis=0).reset_index(drop=True)

df.head()

#Cleaning the dataset to bring some uniformity to the data
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[^a-zA-Z ]", " ", t)
    t = " ".join([w for w in t.split() if w not in stopwords])
    return t

df["clean"] = df["text"].astype(str).apply(clean_text)

# Encode labels
encoder = LabelEncoder()
df["label_enc"] = encoder.fit_transform(df["label"])

texts = df["clean"].tolist()
labels = df["label_enc"].values

#Converting the data into Tokens
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

MAX_LEN = 300
X = pad_sequences(sequences, maxlen=MAX_LEN)
vocab_size = len(tokenizer.word_index) + 1

#Topic modeling of the tokens
tokenized = [nltk.word_tokenize(t) for t in texts]

dictionary = Dictionary(tokenized)
corpus = [dictionary.doc2bow(t) for t in tokenized]

NUM_TOPICS = 10
lda = LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=10)

def get_topic_vector(doc_tokens):
    bow = dictionary.doc2bow(doc_tokens)
    topic_dist = lda.get_document_topics(bow, minimum_probability=0)
    return np.array([t[1] for t in topic_dist])

topic_vectors = np.array([get_topic_vector(doc) for doc in tokenized])

#using Word2Vec to convert the tokens into vectors
w2v = Word2Vec(sentences=tokenized, vector_size=100, min_count=2)

embedding_matrix = np.zeros((vocab_size, 100))
for word, idx in tokenizer.word_index.items():
    if word in w2v.wv:
        embedding_matrix[idx] = w2v.wv[word]

#The Juicy model part - HYBRID MODEL (CNN + BiLSTM + Attention + Topics)
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.w = Dense(1)

    def call(self, inputs):
        scores = self.w(inputs)
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(inputs * weights, axis=1)
        return context

# Inputs
text_input = Input(shape=(MAX_LEN,), name="text_input")
topic_input = Input(shape=(NUM_TOPICS,), name="topic_input")

# Embedding layer
embed = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(text_input)

# CNN + BiLSTM + Attention
cnn = Conv1D(64, 5, activation="relu")(embed)
cnn = MaxPooling1D(2)(cnn)

lstm = Bidirectional(LSTM(64, return_sequences=True))(cnn)
att = AttentionLayer()(lstm)

concat = concatenate([att, topic_input])

dense = Dense(64, activation="relu")(concat)
output = Dense(1, activation="sigmoid")(dense)

model = Model(inputs=[text_input, topic_input], outputs=output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

#Spilting the data into training and testing
X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
    X, topic_vectors, labels, test_size=0.2, random_state=42
)

history = model.fit(
    [X_train, t_train], y_train,
    validation_data=([X_test, t_test], y_test),
    epochs=5,
    batch_size=64
)

#Evaluation
loss, acc = model.evaluate([X_test, t_test], y_test)
print("Test Accuracy:", acc)

# The Explainability
class_names = ["FAKE", "REAL"]   # index = label (0 = fake, 1 = real)

explainer = LimeTextExplainer(class_names=class_names)

def predict_proba(text_list):
    seq = tokenizer.texts_to_sequences(text_list)
    pad = pad_sequences(seq, maxlen=MAX_LEN)

    tokens = [nltk.word_tokenize(t) for t in text_list]
    t_vecs = np.array([get_topic_vector(tok) for tok in tokens])

    preds = model.predict([pad, t_vecs])
    return np.hstack([1-preds, preds])

# Example explainability
idx = 10
exp = explainer.explain_instance(df["clean"][idx], predict_proba, num_features=10)
exp.show_in_notebook()

#Output
def predict_news(text):
    c = clean_text(text)
    seq = tokenizer.texts_to_sequences([c])
    pad = pad_sequences(seq, maxlen=MAX_LEN)

    tok = nltk.word_tokenize(c)
    tvec = np.array([get_topic_vector(tok)])

    p = model.predict([pad, tvec])[0][0]
    print(p)
    return "REAL" if p >= 0.5 else "FAKE"   # label 1 = REAL (True.csv)

print(predict_news("The planet mars blew up today"))
