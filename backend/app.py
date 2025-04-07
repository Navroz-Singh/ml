from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import numpy as np

app = Flask(__name__)
CORS(app)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

vectorizer = load_pickle('models/vectorizer.pkl')
svd = load_pickle('models/svd.pkl')
scaler = load_pickle('models/scaler.pkl')

lr_model = load_pickle('models/logistic_regression.pkl')
svm_model = load_pickle('models/svm_rbf.pkl')
rf_model = load_pickle('models/random_forest.pkl')
ann_model = load_pickle('models/neural_network.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    headline = data.get('headline', '')
    body = data.get('body', '')

    if not headline or not body:
        return jsonify({'error': 'Headline and Body are required'}), 400

    # Combine text
    text = headline + ' ' + body
    tfidf_vector = vectorizer.transform([text])
    svd_vector = svd.transform(tfidf_vector)
    scaled_vector = scaler.transform(svd_vector)

    predictions = {
        'Logistic Regression': int(lr_model.predict(svd_vector)[0]),
        'RBF SVM': int(svm_model.predict(scaled_vector)[0]),
        'Random Forest': int(rf_model.predict(tfidf_vector)[0]),
        'Neural Network': int(ann_model.predict(scaled_vector)[0]),
    }

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
