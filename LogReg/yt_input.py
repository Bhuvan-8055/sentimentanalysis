import joblib
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
import re

# Load trained model and vectorizer
model = joblib.load("comments_sentiment_model.pkl")
vectorizer = joblib.load("comments_tfidf_vectorizer.pkl")

# Load emoji sentiment ranking dataset
emoji_df = pd.read_excel("emoji_sentiment_ranking.xlsx")
emoji_dict = dict(zip(emoji_df["Char"], emoji_df["Sentiment score"]))

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to compute emoji sentiment score
def get_emoji_sentiment(text):
    emojis_in_text = [char for char in text if char in emoji_dict]  # Extract emojis
    if not emojis_in_text:
        return 0  # No emoji = Neutral sentiment
    scores = [emoji_dict.get(char, 0) for char in emojis_in_text]
    return sum(scores) / len(scores)

# Function to compute text sentiment score
def get_text_sentiment(text):
    scores = analyzer.polarity_scores(str(text))  # Convert to string (avoid NaN errors)
    return scores["compound"]  # Returns a score between -1 and 1

# Function to classify sentiment using both text and emoji sentiment
def classify_sentiment(text, model, vectorizer):
    text_sentiment = get_text_sentiment(text)  # Get text sentiment score
    emoji_sentiment = get_emoji_sentiment(text)  # Get emoji sentiment score

    # Combine text and emoji sentiment (equal weight)
    final_score = (text_sentiment + emoji_sentiment) / 2  

    # Predict sentiment using ML model
    text_vector = vectorizer.transform([text])
    ml_prediction = model.predict(text_vector)[0]

    # Adjust final sentiment label
    if final_score >= 0.05:
        return "Positive"
    elif final_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# ========= 🔹 Test with Custom Inputs =========
while True:
    user_input = input("\nEnter a comment (or type 'exit' to stop): ")
    if user_input.lower() == "exit":
        break
    
    sentiment = classify_sentiment(user_input, model, vectorizer)
    print(f"🧐 Sentiment: {sentiment}")
