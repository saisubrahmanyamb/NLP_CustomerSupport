from flask import Flask, render_template, request
import joblib
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load trained model and vectorizer
model = joblib.load("model/customer_support_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

app = Flask(__name__)

# Text cleaning function
def clean_text(text):

    text = str(text)

    text = text.lower()

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Automated replies
replies = {

    "Product setup": "Please follow the setup instructions provided with your product.",

    "Peripheral compatibility": "Please verify whether your device supports the required peripherals.",

    "Network problem": "Please check your internet connection and router settings.",

    "Account access": "Please reset your password or verify your login credentials.",

    "Data loss": "We recommend restoring your backup and contacting support immediately.",

    "Payment issue": "Please retry the transaction or contact your bank.",

    "Refund request": "Your refund request is being reviewed by our support team.",

    "Battery life": "Please optimize device settings and check for firmware updates.",

    "Installation support": "Our technical team will guide you through the installation process."
}

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    reply = None
    user_message = None

    if request.method == "POST":

        user_message = request.form["message"]

        cleaned = clean_text(user_message)

        vectorized = vectorizer.transform([cleaned])

        predicted_category = model.predict(vectorized)[0]

        probabilities = model.predict_proba(vectorized)

        confidence_score = round(
            max(probabilities[0]) * 100,
            2
        )

        reply = replies.get(
            predicted_category,
            "Our support team will contact you shortly."
        )

        prediction = predicted_category
        confidence = confidence_score

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        reply=reply,
        user_message=user_message
    )

if __name__ == "__main__":
    app.run(debug=True)