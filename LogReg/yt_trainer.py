import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ========= 🔹 STEP 1: Load Final Dataset =========
df = pd.read_csv("yt2_final_sentiment.csv", encoding="utf-8-sig")

# ========= 🔹 STEP 2: Fix Missing Values =========
df["Cleaned_Comment"] = df["Cleaned_Comment"].fillna("")  # Replace NaN with empty strings

# Convert text to numerical features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["Cleaned_Comment"])  
y = df["Final_Sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model & vectorizer
import joblib
joblib.dump(model, "comments_sentiment_model.pkl")
joblib.dump(vectorizer, "comments_tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved as 'sentiment_model.pkl' & 'tfidf_vectorizer.pkl'")
