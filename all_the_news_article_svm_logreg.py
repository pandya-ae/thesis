import pandas as pd

# Load dataset from a CSV file
data_path = '/content/drive/MyDrive/dataset/all-the-news-combined.csv'
data = pd.read_csv(data_path)
data.dropna(subset=['author', 'content'], inplace=True)  # Remove rows with missing 'author' or 'content'

# Filter data to include only authors with a significant number (400 or more) of articles
author_counts = data['author'].value_counts()
frequent_authors = author_counts[author_counts >= 400].index

# List of grouped authors (not individuals) to be excluded from the analysis
authors_to_exclude = [
    "Associated Press",
    "Breitbart Jerusalem",
    "Breitbart London",
    "Breitbart News",
    "Editorial Board",
    "Fox News",
    "NPR Staff",
    "Post Editorial Board",
    "Post Staff Report",
    "Reuters",
    "The Editors"
]

# Apply filters to the dataset
data = data[data['author'].isin(frequent_authors) & ~data['author'].isin(authors_to_exclude)]

# Display the names of qualified authors
qualified_authors = data['author'].unique()
print("Qualified Authors:")
for author in qualified_authors:
    print(author)

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text by removing digits, punctuation, and stop words, and applying lemmatization
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return ' '.join(lemmatizer.lemmatize(word) for word in words if word not in stop_words)

# Apply preprocessing to each content entry in the dataset
data['processed_content'] = data['content'].apply(preprocess_text)

from sklearn.feature_extraction.text import TfidfVectorizer

# Prepare for TF-IDF vectorization
all_words = set()
data['processed_content'].str.split().apply(all_words.update)
max_features = len(all_words)  # Set the maximum number of features as the total number of unique words

# Vectorize the processed text using TF-IDF method
tfidf_vectorizer = TfidfVectorizer(max_features=max_features, min_df=3, max_df=0.9, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(data['processed_content'])
y = data['author']

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Encode author names into categorical labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Initialize and train an SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Initialize and train a Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)
y_pred_log_reg = log_reg_model.predict(X_test)

# Print classification reports for SVM and Logistic Regression
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=encoder.classes_))

print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg, target_names=encoder.classes_))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    labels = [f'{i} - {cls}' for i, cls in enumerate(classes)]
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    return cm

# Function to evaluate the model and print performance metrics
def evaluate_model(name, y_true, y_pred, classes):
    print(f"{name} Model Evaluation")
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    precision = precision_score(y_true, y_pred, average='macro')
    print("Precision:", precision)
    recall = recall_score(y_true, y_pred, average='macro')
    print("Recall:", recall)
    f1 = f1_score(y_true, y_pred, average='macro')
    print("F1 Score:", f1)
    cm = plot_confusion_matrix(y_true, y_pred, classes)
    print("Detailed Confusion Matrix Breakdown:")
    for i in range(len(classes)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        print(f"Author {i} - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

# Evaluate and print the performance of the SVM and Logistic Regression models
evaluate_model("SVM", y_test, y_pred_svm, encoder.classes_)
evaluate_model("Logistic Regression", y_test, y_pred_log_reg, encoder.classes_)

pip install joblib

from joblib import dump

# Save the SVM model
dump(svm_model, '/content/svm_model.joblib')

# Save the Logistic Regression model
dump(log_reg_model, '/content/log_reg_model.joblib')

# Save the TF-IDF Vectorizer
dump(tfidf_vectorizer, '/content/tfidf_vectorizer.joblib')

# Save the LabelEncoder
dump(encoder, '/content/label_encoder.joblib')

import zipfile

# Create a zip file containing all necessary files
with zipfile.ZipFile('/content/model_files.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write('/content/svm_model.joblib', arcname='svm_model.joblib')
    zipf.write('/content/log_reg_model.joblib', arcname='log_reg_model.joblib')
    zipf.write('/content/tfidf_vectorizer.joblib', arcname='tfidf_vectorizer.joblib')
    zipf.write('/content/label_encoder.joblib', arcname='label_encoder.joblib')