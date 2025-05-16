import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ========= 🔹 STEP 1: Load Dataset =========
df = pd.read_csv("youtube_comments_with_emoji_scores.csv", encoding="utf-8-sig")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# ========= 🔹 STEP 2: Apply VADER for Text Sentiment =========
def get_text_sentiment(text):
    scores = analyzer.polarity_scores(str(text))  # Convert to string (avoid NaN errors)
    return scores["compound"]  # Returns a score between -1 and 1

# Apply text sentiment analysis
df["Text_Sentiment_Score"] = df["Cleaned_Comment"].apply(get_text_sentiment)

# ========= 🔹 STEP 3: Combine Text & Emoji Sentiments =========
def classify_sentiment(text_score, emoji_score):
    # Combine text and emoji sentiment (equal weight)
    final_score = (text_score + emoji_score) / 2  

    # Assign sentiment labels
    if final_score >= 0.05:
        return "Positive"
    elif final_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Compute final sentiment label
df["Final_Sentiment"] = df.apply(lambda row: classify_sentiment(row["Text_Sentiment_Score"], row["Emoji_Sentiment_Score"]), axis=1)

# ========= 🔹 STEP 4: Save Final Dataset =========
df.to_csv("youtube_final_sentiment.csv", index=False, encoding="utf-8-sig")

print("✅ Final sentiment labels saved in 'youtube_final_sentiment.csv'")

# Preview results
print(df[["Cleaned_Comment", "Text_Sentiment_Score", "Emoji_Sentiment_Score", "Final_Sentiment"]].head())
