from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
df = pd.read_csv("youtube_comments_with_emoji_scores.csv", encoding="utf-8")
def get_text_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores["compound"] >= 0.05:
        return "Positive"
    elif scores["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis on cleaned comments
df["Text_Sentiment"] = df["Cleaned_Comment"].apply(get_text_sentiment)

# Preview results
print(df[["Cleaned_Comment", "Text_Sentiment"]].head())

# Save labeled dataset
df.to_csv("youtube_comments_with_sentiment.csv", index=False, encoding="utf-8-sig")

print("✅ Labeled dataset saved as 'youtube_comments_with_sentiment.csv'")
