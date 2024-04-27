import pandas as pd

# Load dataset and GloVe word embedding from CSV file
data_path = '/content/drive/MyDrive/dataset/all-the-news-combined.csv'
glove_path = '/content/drive/MyDrive/glove/glove.6B.300d.txt'
data = pd.read_csv(data_path)
data.dropna(subset=['author', 'content'], inplace=True)  # Remove entries with missing author or content

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

# Filter out the authors in the exclusion list
data = data[data['author'].isin(frequent_authors) & ~data['author'].isin(authors_to_exclude)]

# Display the names of qualified authors
qualified_authors = data['author'].unique()
print("Qualified Authors:")
for author in qualified_authors:
    print(author)

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')  # Download the Punkt tokenizer models

# Tokenize the content of each article into sentences using NLTK's sent_tokenize
data['sentences'] = data['content'].apply(sent_tokenize)

# Explode the DataFrame to separate each sentence into its own row, while keeping the associated author the same
# This operation transforms each list of sentences into separate rows, duplicating the other column values as needed.
data = data.explode('sentences').reset_index(drop=True)

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('stopwords')

# Function to preprocess text by removing digits, punctuation, and stop words, and applying lemmatization
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return ' '.join(lemmatizer.lemmatize(word) for word in words if word not in stop_words)

# Apply preprocessing to the content of each article
data['processed_content'] = data['sentences'].apply(preprocess_text)

import numpy as np

# Function to load GloVe embeddings from file
def load_glove_embeddings(path):
    embeddings_index = {}
    dimension = None
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            if dimension is None:
                dimension = len(values) - 1  # Set the dimensionality of the vectors
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index, dimension

# Load GloVe embeddings
glove_embeddings, embedding_dim = load_glove_embeddings(glove_path)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize and fit tokenizer on the processed content
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['processed_content'])
sequences = tokenizer.texts_to_sequences(data['processed_content'])

# Determine the optimal length of sequences for padding
sequence_lengths = [len(seq) for seq in sequences]
maxlen = int(np.percentile(sequence_lengths, 95))  # Using the 95th percentile to determine maxlen

# Pad sequences to ensure consistent length
X = pad_sequences(sequences, maxlen=maxlen)

# Prepare the embedding matrix by mapping GloVe vectors to the corresponding words in our tokenizer
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Encode author names into categorical labels
encoder = LabelEncoder()
y = encoder.fit_transform(data['author'])

# Split data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, GRU, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# Define the LSTM model architecture
model_lstm = Sequential([
    Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True),
    SpatialDropout1D(0.2),
    LSTM(128, return_sequences=True, dropout=0.2),
    Dropout(0.25),
    LSTM(128, return_sequences=True, dropout=0.2),
    Dropout(0.25),
    LSTM(64, dropout=0.2),
    Dropout(0.2),
    Dense(len(set(y)), activation='softmax')
])

# Compile the LSTM model with defined optimizer and loss function
optimizer = RMSprop(learning_rate=0.001)
model_lstm.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping to halt training when validation loss ceases to decrease
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)

# Train the LSTM model with early stopping
history_lstm = model_lstm.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the LSTM model on the test data
test_loss_lstm, test_acc_lstm = model_lstm.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {test_loss_lstm}, Test Accuracy: {test_acc_lstm}")

# Predict labels for the test set and generate a classification report
y_pred_lstm = model_lstm.predict(X_test)
y_pred_classes_lstm = np.argmax(y_pred_lstm, axis=1) # Convert probabilities to class labels
print("LSTM Classification Report:")
print(classification_report(y_test, y_pred_classes_lstm, target_names=encoder.classes_))

# Define the GRU model architecture
model_gru = Sequential([
    Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True),
    SpatialDropout1D(0.2),
    GRU(128, return_sequences=True, dropout=0.2),
    GRU(128, return_sequences=True, dropout=0.2),
    GRU(64, dropout=0.2),
    Dropout(0.2),
    Dense(len(set(y)), activation='softmax')
])

# Compile the GRU model with defined optimizer and loss function
optimizer = RMSprop(learning_rate=0.001)
model_gru.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping to halt training when validation loss ceases to decrease
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)

# Train the GRU model with early stopping
history_gru = model_gru.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the GRU model on the test data
test_loss_gru, test_acc_gru = model_gru.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {test_loss_gru}, Test Accuracy: {test_acc_gru}")

# Predict labels for the test set and generate a classification report
y_pred_gru = model_gru.predict(X_test)
y_pred_classes_gru = np.argmax(y_pred_gru, axis=1) # Convert probabilities to class labels
print("GRU Classification Report:")
print(classification_report(y_test, y_pred_classes_gru, target_names=encoder.classes_))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Function to plot training history for accuracy and loss
def plot_history(history, title):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(title + ' - Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title + ' - Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    labels = [f'{i} - {class_name}' for i, class_name in enumerate(classes)]

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return cm

# Function to evaluate model performance
def evaluate_model(name, y_true, y_pred, classes, history):
    plot_history(history, name)  # Plot the training and validation history
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"{name} Model Evaluation")
    print("Accuracy: {:.2f}".format(accuracy))
    print("Macro Precision: {:.2f}".format(precision))
    print("Macro Recall: {:.2f}".format(recall))
    print("Macro F1 Score: {:.2f}".format(f1))

    cm = plot_confusion_matrix(y_true, y_pred, classes, name)  # Plot the confusion matrix

    # Detailed breakdown of the confusion matrix
    print(f"Confusion Matrix Breakdown for {name}:")
    for i in range(len(classes)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        print(f"Author {i} - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

# Evaluating LSTM model
evaluate_model("LSTM", y_test, y_pred_classes_lstm, encoder.classes_, history_lstm)

# Evaluating GRU model
evaluate_model("GRU", y_test, y_pred_classes_gru, encoder.classes_, history_gru)

# Save the LSTM and GRU models to the specified directory
model_lstm.save('/content/model_lstm.h5')  # saves the model in HDF5 format
model_gru.save('/content/model_gru.h5')

import pickle

# Save the tokenizer
with open('/content/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the LabelEncoder
with open('/content/encoder.pickle', 'wb') as handle:
    pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

import zipfile

# Create a zip file containing the models and other necessary files
with zipfile.ZipFile('/content/model_files.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write('/content/model_lstm.h5', arcname='model_lstm.h5')
    zipf.write('/content/model_gru.h5', arcname='model_gru.h5')
    zipf.write('/content/tokenizer.pickle', arcname='tokenizer.pickle')
    zipf.write('/content/encoder.pickle', arcname='encoder.pickle')