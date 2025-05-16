import pandas as pd
import emoji
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

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

# Load preprocessed Reddit & Twitter data
reddit_df = pd.read_csv(r"C:\Users\sribh\OneDrive\Desktop\Up_reddit_cleaned.csv")
twitter_df = pd.read_csv(r"C:\Users\sribh\OneDrive\Desktop\Up_twitter_cleaned.csv")

# Fill NaN values in text columns
reddit_df["clean_comment"] = reddit_df["clean_comment"].fillna("")
twitter_df["clean_text"] = twitter_df["clean_text"].fillna("")

# Function to process rows with a progress counter
def process_with_counter(df, column_name):
    total_rows = len(df)
    processed_rows = 0
    sentiment_features = []

    for text in df[column_name]:
        sentiment_features.append(extract_emoji_sentiment(text))
        processed_rows += 1
        print(f"✅ Processing Row {processed_rows}/{total_rows}", end="\r")  # Dynamic progress update

    return pd.DataFrame(sentiment_features, columns=["positive_emoji", "negative_emoji", "neutral_emoji"])

# Extract emoji sentiment features with progress tracking
print("🔹 Extracting emoji sentiment features for Reddit data...")
reddit_df[["positive_emoji", "negative_emoji", "neutral_emoji"]] = process_with_counter(reddit_df, "clean_comment")

print("\n🔹 Extracting emoji sentiment features for Twitter data...")
twitter_df[["positive_emoji", "negative_emoji", "neutral_emoji"]] = process_with_counter(twitter_df, "clean_text")

# Combine text data
text_data = pd.concat([reddit_df["clean_comment"], twitter_df["clean_text"]], ignore_index=True)

# Fill NaN values before TF-IDF processing
text_data = text_data.fillna("")

# Load labels
labels = pd.concat([reddit_df["category"], twitter_df["category"]], ignore_index=True)

# Convert labels to numerical format
label_mapping = {label: idx for idx, label in enumerate(labels.unique())}
labels = labels.map(label_mapping)

# TF-IDF Vectorization
print("\n🔹 Applying TF-IDF vectorization...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)

# Convert emoji sentiment features to DataFrame
emoji_features = pd.concat([
    reddit_df[["positive_emoji", "negative_emoji", "neutral_emoji"]],
    twitter_df[["positive_emoji", "negative_emoji", "neutral_emoji"]]
], ignore_index=True)

# Convert emoji features to sparse matrix and combine with TF-IDF
final_features = hstack([tfidf_matrix, emoji_features])

# Split data
X_train, X_test, y_train, y_test = train_test_split(final_features, labels, test_size=0.2, random_state=42)

# Train SVM model
print("\n🔹 Training SVM model...")
svm_model = SVC(kernel="linear", C=1.0)  # Linear kernel is best for text classification
svm_model.fit(X_train, y_train)

# Predictions
print("\n🔹 Making predictions...")
y_pred = svm_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ SVM Model Training Complete! Accuracy: {accuracy:.4f}")
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained SVM model & vectorizer
with open(r"C:\Users\sribh\OneDrive\Desktop\svm_with_emoji.pkl", "wb") as f:
    pickle.dump(svm_model, f)

with open(r"C:\Users\sribh\OneDrive\Desktop\tfidf_vectorizer_with_emoji.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

print(f"\n🎯 SVM model (with emoji sentiment) saved as 'svm_with_emoji.pkl'")
