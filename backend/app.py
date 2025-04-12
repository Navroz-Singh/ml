from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

# Load tokenizer
with open('models/tokenizer.json', 'r') as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_data)

max_sequence_length = 200

base_model = load_model('models/base_model.h5')
improved_model = load_model('models/improved_model.h5')

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(stemmer.stem(w)) for w in words]
    return " ".join(words)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    body = data.get('body')

    if not body:
        return jsonify({'error': 'Body is required'}), 400

    cleaned_text = clean_text(body)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=max_sequence_length)

    base_pred = base_model.predict(padded)[0][0]
    improved_pred = improved_model.predict(padded)[0][0]

    predictions = {
        'Base Model': int(base_pred > 0.5),
        'Improved Model': int(improved_pred > 0.5)
    }

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
