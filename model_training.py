import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv("dataset/labeled_data.csv")

# Check column names
print("Dataset Columns:", data.columns)

# Ensure the correct column name for text
text_column = "tweet" if "tweet" in data.columns else data.columns[0]

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
data['clean_text'] = data[text_column].astype(str).apply(clean_text)

# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
text_features = vectorizer.fit_transform(data['clean_text']).toarray()

# Labels
y = data['class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(text_features, y, test_size=0.3, random_state=42)

# Models to Compare
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naïve Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# Train and Evaluate Models
best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

print(f"Best Model: {best_model.__class__.__name__} with Accuracy: {best_accuracy:.4f}")

# Save the Best Model and Vectorizer
pickle.dump(best_model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Best model and vectorizer saved successfully!")
