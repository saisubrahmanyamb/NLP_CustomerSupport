import pandas as pd
import string
import nltk
import joblib
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("dataset/customer_support_tickets.csv")

# Keep important columns only
df = df[['Ticket Subject', 'Ticket Description', 'Ticket Type']]

# Remove missing values
df = df.dropna()

# Combine subject + description
df['combined_text'] = (
    df['Ticket Subject'] + " " + df['Ticket Description']
)

# Text cleaning function
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove placeholders
    text = re.sub(r'\{.*?\}', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Tokenization
    words = word_tokenize(text)

    # Stopwords removal
    stop_words = set(stopwords.words('english'))

    words = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)

# Apply preprocessing
df['cleaned_text'] = df['combined_text'].apply(clean_text)

# Features and labels
X = df['cleaned_text']
y = df['Ticket Subject']

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_vectorized = vectorizer.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression model
model = LogisticRegression(max_iter=2000)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(
    model,
    "model/customer_support_model.pkl"
)

# Save vectorizer
joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer.pkl"
)

print("\nModel and vectorizer saved successfully")