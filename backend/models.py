import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import TruncatedSVD

data = pd.read_csv('data.csv')
print(data.head())
print(f"Data shape: {data.shape}")
print(f"Missing values: {data.isnull().sum()}")
data = data.dropna()
print(f"Class distribution: {data['Label'].value_counts()}")

data['text'] = data['Headline'] + ' ' + data['Body']

X = data['text']
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    use_idf=True,
    norm='l2'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

svd = TruncatedSVD(n_components=300, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

print(f"Reduced dimensions: {X_train_svd.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_svd)
X_test_scaled = scaler.transform(X_test_svd)

lr_model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='liblinear',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
lr_model.fit(X_train_svd, y_train)
lr_preds = lr_model.predict(X_test_svd)

print("Enhanced Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
print(classification_report(y_test, lr_preds))

svm_model = SVC(
    C=10.0,
    kernel='rbf',
    gamma='auto',
    probability=True,
    class_weight='balanced',
    random_state=42
)
svm_model.fit(X_train_scaled, y_train)
svm_preds = svm_model.predict(X_test_scaled)

print("Enhanced RBF SVM Results:")
print(f"Accuracy: {accuracy_score(y_test, svm_preds):.4f}")
print(classification_report(y_test, svm_preds))

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train_tfidf, y_train)
rf_preds = rf_model.predict(X_test_tfidf)

print("Enhanced Random Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(classification_report(y_test, rf_preds))

ann_model = MLPClassifier(
    hidden_layer_sizes=(200, 100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=64,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
ann_model.fit(X_train_scaled, y_train)
ann_preds = ann_model.predict(X_test_scaled)

print("Enhanced Neural Network Results:")
print(f"Accuracy: {accuracy_score(y_test, ann_preds):.4f}")
print(classification_report(y_test, ann_preds))

models = {
    'Logistic Regression': lr_preds,
    'RBF SVM': svm_preds,
    'Random Forest': rf_preds,
    'Neural Network': ann_preds
}

for model_name, predictions in models.items():
    acc = accuracy_score(y_test, predictions)
    print(f"{model_name} Accuracy: {acc:.4f}")

if hasattr(rf_model, 'feature_importances_'):
    feature_names = vectorizer.get_feature_names_out()
    indices = np.argsort(rf_model.feature_importances_)[::-1][:20]
    
    print("\nTop 20 features for rumor classification:")
    for i in indices:
        print(f"{feature_names[i]}: {rf_model.feature_importances_[i]:.4f}")

os.makedirs('saved_models', exist_ok=True)
os.makedirs('models', exist_ok=True)

with open('saved_models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('saved_models/svm_rbf.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

with open('saved_models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('saved_models/neural_network.pkl', 'wb') as f:
    pickle.dump(ann_model, f)

with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('models/svd.pkl', 'wb') as f:
    pickle.dump(svd, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
