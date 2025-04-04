import os
import pandas as pd
import numpy as np
import re
import pickle
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


# Ensure TensorFlow version compatibility
print(f"Using TensorFlow Version: {tf.__version__}")  # Should print 2.12.0

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load both datasets
data1 = pd.read_csv("dataset/labeled_data.csv")  # Original dataset
data2 = pd.read_csv("dataset/generated_dataset.csv")  # Synthetic dataset

# Combine datasets
data = pd.concat([data1, data2], ignore_index=True)

# Define stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Apply text preprocessing
data['clean_text'] = data['tweet'].astype(str).apply(clean_text)

# Tokenization and Padding
max_words = 20000  # Vocabulary size
max_len = 100  # Max sequence length
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(data['clean_text'])
sequences = tokenizer.texts_to_sequences(data['clean_text'])
X = pad_sequences(sequences, maxlen=max_len)
y = data['class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Compute Class Weights (Handles Imbalanced Data)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Define LSTM Model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)),  
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 classes (Hate, Offensive, Neutral)
])

# Compile Model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model Save Paths
model_path_keras = "lstm_hatespeech_model.keras"
model_path_h5 = "lstm_hatespeech_model.h5"

# 🔹 Delete previous models safely before saving
for path in [model_path_keras, model_path_h5]:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass  # Ignore if file doesn't exist

# Train Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weight_dict)

# Evaluate Model
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ Save Model in Keras Format (Recommended for Keras 3)
model.save(model_path_keras)  

# ✅ Save Model in HDF5 Format (Only if Needed)
model.save(model_path_h5)

# Save Tokenizer
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))

print(f"✅ Model saved as {model_path_keras} and {model_path_h5}, tokenizer saved as tokenizer.pkl")

# 🔹 Ensure Model Loading Works Correctly in app.py
try:
    model = load_model(model_path_keras)
    print("✅ Model loaded successfully from .keras!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

try:
    model = load_model(model_path_h5)
    print("✅ Model loaded successfully from .h5!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
