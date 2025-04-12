import numpy as np
import pandas as pd
import re
import nltk
import pickle as pkl
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout


nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(stemmer.stem(w)) for w in words]
    return " ".join(words)

data_path = 'data.csv'
df = pd.read_csv(data_path)
df.dropna(subset=['text', 'label'], inplace=True)
df = df[df['label'].astype(str).str.strip().isin(['0', '1'])]
df['label'] = df['label'].astype(int)
df.dropna(subset=['text', 'label'], inplace=True)

print("Pre-processing texts ...")
df['clean_text'] = df['text'].apply(clean_text)

texts = df['clean_text'].values
labels = df['label'].values

max_num_words = 10000
tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

max_sequence_length = 200
X = pad_sequences(sequences, maxlen=max_sequence_length)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

embedding_dim = 128
model = Sequential()
model.add(Embedding(input_dim=max_num_words, output_dim=embedding_dim, input_shape=(max_sequence_length,)))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

batch_size = 16
epochs = 8

history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy:.4f}")

y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

filename = 'base_model.pkl'
with open(filename, 'wb') as file:
    pkl.dump(model, file)
