import pickle
import pandas as pd
import emoji
from scipy.sparse import hstack

# Load trained SVM model
with open(r"C:\Users\sribh\OneDrive\Desktop\svm_with_emoji.pkl", "rb") as f:
    svm_model = pickle.load(f)

# Load TF-IDF vectorizer
with open(r"C:\Users\sribh\OneDrive\Desktop\tfidf_vectorizer_with_emoji.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Load emoji sentiment dataset
emoji_df = pd.read_excel(r"C:\Users\sribh\OneDrive\Desktop\emoji_sentiment_ranking.xlsx")

# Convert Sentiment score column to numeric
emoji_df["Sentiment score"] = pd.to_numeric(emoji_df["Sentiment score"], errors="coerce")

# Drop NaN values
emoji_df = emoji_df.dropna(subset=["Sentiment score"])

# Create an emoji sentiment dictionary
emoji_sentiment_dict = {row["Char"]: float(row["Sentiment score"]) for _, row in emoji_df.iterrows()}

# Function to extract emoji sentiment features from text
def extract_emoji_sentiment(text):
    if not isinstance(text, str):
        return 0, 0, 0

    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for char in text:
        if char in emoji_sentiment_dict:
            score = emoji_sentiment_dict[char]
            if score > 0:
                positive_count += 1
            elif score < 0:
                negative_count += 1
            else:
                neutral_count += 1

    return positive_count, negative_count, neutral_count

# Function to predict sentiment of input text
def predict_sentiment(user_text):
    # Convert input text into TF-IDF features
    user_text_tfidf = tfidf_vectorizer.transform([user_text])

    # Extract emoji sentiment features
    pos_emoji, neg_emoji, neu_emoji = extract_emoji_sentiment(user_text)

    # Convert emoji sentiment features to a sparse matrix
    emoji_features = pd.DataFrame([[pos_emoji, neg_emoji, neu_emoji]])
    emoji_features_sparse = hstack([user_text_tfidf, emoji_features])

    # Make prediction
    predicted_label = svm_model.predict(emoji_features_sparse)[0]

    # Mapping labels back to original categories
    label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Adjust based on your dataset labels
    predicted_sentiment = label_mapping.get(predicted_label, "Unknown")

    return predicted_sentiment

# Interactive loop for user input
while True:
    user_input = input("Enter a sentence to analyze (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    result = predict_sentiment(user_input)
    print(f"🔹 Predicted Sentiment: {result}")
