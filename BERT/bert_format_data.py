import pandas as pd
import torch
from transformers import BertTokenizer
import time

# Load cleaned datasets
reddit_df = pd.read_csv(r"C:\Users\sribh\OneDrive\Desktop\Up_reddit_cleaned.csv")
twitter_df = pd.read_csv(r"C:\Users\sribh\OneDrive\Desktop\Up_twitter_cleaned.csv")

# Load emoji sentiment dataset
emoji_df = pd.read_excel(r"C:\Users\sribh\OneDrive\Desktop\emoji_sentiment_ranking.xlsx")

# Convert Sentiment score column to numeric
emoji_df["Sentiment score"] = pd.to_numeric(emoji_df["Sentiment score"], errors="coerce")

# Drop NaN values
emoji_df = emoji_df.dropna(subset=["Sentiment score"])

# Create an emoji sentiment dictionary
emoji_sentiment_dict = {row["Char"]: float(row["Sentiment score"]) for _, row in emoji_df.iterrows()}

# Function to extract emoji sentiment features with progress tracking
def extract_emoji_sentiment(texts):
    positive_counts, negative_counts, neutral_counts = [], [], []
    total_rows = len(texts)

    print("\n🔹 Extracting emoji sentiment features...")
    start_time = time.time()

    for i, text in enumerate(texts, 1):
        if not isinstance(text, str):
            pos, neg, neu = 0, 0, 0
        else:
            pos = sum(1 for char in text if char in emoji_sentiment_dict and emoji_sentiment_dict[char] > 0)
            neg = sum(1 for char in text if char in emoji_sentiment_dict and emoji_sentiment_dict[char] < 0)
            neu = sum(1 for char in text if char in emoji_sentiment_dict and emoji_sentiment_dict[char] == 0)

        positive_counts.append(pos)
        negative_counts.append(neg)
        neutral_counts.append(neu)

        if i % 5000 == 0 or i == total_rows:  # Print progress every 5000 rows
            elapsed = time.time() - start_time
            print(f"✅ Processed {i}/{total_rows} rows ({elapsed:.2f} sec elapsed)", end="\r")

    print(f"\n✅ Emoji feature extraction complete! Processed {total_rows} rows.")
    return positive_counts, negative_counts, neutral_counts

# Combine datasets
df = pd.concat([reddit_df, twitter_df], ignore_index=True)

# Select text and labels
texts = df["clean_comment"].fillna("").tolist()  # Use 'clean_text' for Twitter if needed
labels = df["category"].tolist()  # Assuming 'category' contains sentiment labels

# Convert labels to numerical format
label_mapping = {label: idx for idx, label in enumerate(set(labels))}
labels = [label_mapping[label] for label in labels]

# Extract emoji sentiment features (with progress tracking)
pos_emoji, neg_emoji, neu_emoji = extract_emoji_sentiment(texts)

# Convert to DataFrame
emoji_features_df = pd.DataFrame({"positive_emoji": pos_emoji, "negative_emoji": neg_emoji, "neutral_emoji": neu_emoji})

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize text with progress tracking
print("\n🔹 Tokenizing text for BERT...")
start_time = time.time()
tokens = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

elapsed = time.time() - start_time
print(f"✅ Tokenization complete! Processed {len(texts)} rows in {elapsed:.2f} sec.")

# Convert labels to tensor
labels_tensor = torch.tensor(labels)

# Convert emoji features to tensor
emoji_tensor = torch.tensor(emoji_features_df.values, dtype=torch.float)

# Save processed data for training
torch.save(tokens, r"C:\Users\sribh\OneDrive\Desktop\bert_tokenized_data.pt")
torch.save(labels_tensor, r"C:\Users\sribh\OneDrive\Desktop\bert_labels.pt")
torch.save(emoji_tensor, r"C:\Users\sribh\OneDrive\Desktop\bert_emoji_features.pt")

print("\n✅ Data Preparation Complete!")
print(f"🔹 Tokenized data saved as 'bert_tokenized_data.pt'")
print(f"🔹 Labels saved as 'bert_labels.pt'")
print(f"🔹 Emoji sentiment features saved as 'bert_emoji_features.pt'")
