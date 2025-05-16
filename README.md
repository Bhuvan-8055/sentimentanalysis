# Sentiment Analysis
# Abstract
This project enhances sentiment analysis on noisy social media text by combining text features with emoji sentiment scores. YouTube comments are cleaned and analyzed using VADER and emoji sentiment data. A Logistic Regression model classifies sentiments as Positive, Negative, or Neutral for better interpretation.
# Problem Statement
Many people use social media platforms like YouTube to share opinions through comments. These comments often include slang, spelling mistakes, and emojis, which makes it difficult for normal sentiment analysis models to understand the true meaning. 
This project aims to solve that problem by improving how we detect emotions and opinions in noisy and informal text using both words and emojis together.
# Review Of Literature
VADER: A Rule-based Model for Sentiment Analysis of Social Media Text
-> Hutto, C. & Gilbert, E. (2014)
-> - Easy to use and interpretable - Performs well on social media text with emojis, slang, and punctuation.
-> Limited in capturing context and sarcasm.
Sentiment of Emojis
->Novak, P. K. (2015)
->- Adds sentiment signals from emojis. - Enhances emotion detection in informal communication.
-> Limited emoji lexicon and cultural interpretation issues.
Enriching Word Vectors with Subword Information
-> Bojanowski, P. et al. (2017)
-> - Improves vector representation for rare and misspelled words.- Useful in multilingual scenarios.
-> Computationally heavier than traditional word embeddings.
# Methodology
1.Data Collection:We collected noisy and informal text from YouTube comments. These comments often contain slang, typos, and emojis. <br>
2.Text Preprocessing:We cleaned the text by removing URLs, special characters, and user mentions. All text was converted to lowercase for consistency.<br>
3.Normalization:Slang and abbreviations were expanded using a custom dictionary (ex: “u” → “you”). Typos were fixed using basic spell correction tools.<br>
4.Emoji Sentiment Mapping:Emojis in the comments were matched with their sentiment scores using an emoji sentiment dataset.<br>
5.Feature Extraction:Text features were extracted using TF-IDF, and sentiment scores were calculated using the VADER tool.<br>
6.Model Training:We used a Logistic Regression model to classify each comment as Positive, Negative, or Neutral based on both text and emoji signals.
# Implementation Details
 Pandas: Used to load, clean, and manage the dataset (YouTube comments).<br>
 Scikit-learn: Used for TF-IDF feature extraction and Logistic Regression model training.<br>
 TF-IDF: Converts cleaned text into numerical vectors so that the model can understand it.<br>
 VADER: Calculates sentiment score of the text using a rule-based method.<br>
 Emoji Sentiment Dataset: Provides predefined sentiment scores for emojis to include emotional context.<br>
 Logistic Regression: A simple and fast machine learning model used to classify sentiments.<br>
 Matplotlib : Used to visualize results like sentiment distribution and performance metrics.<br>
# Results and Discussions
The model achieved around 79% accuracy. It performed best on positive comments and slightly lower on neutral ones. Including emoji sentiment and slang handling helped improve prediction quality for informal text.

Classification Report:
Class         Precision      Recall      F1-score       Support 
Positive         0.82         0.82         0.82          17574 
Neutral          0.79         0.54         0.64          8765
Negative         0.75         0.90         0.82          13705

Accuracy: 0.787
Macro avg        0.79         0.75         0.76          40046
Weighted avg     0.79         0.79         0.78          40046
# Conclusion
This project successfully improves sentiment analysis on noisy and informal text by combining text features with emoji sentiment scores. By cleaning the data, handling slang, and mapping emojis to sentiments, the model becomes more accurate and better suited for social media comments.The final system effectively classifies comments as Positive, Negative, or Neutral and performs well on real-world, unstructured data.
# References
1. Hutto C Gilbert, E. (2014) VADER: A Rule-based Model for Sentiment Analysis of Social Media Text. <br>
2. Novak, P. K  (2015). Sentiment of Emojis<br>
3. Bojanowski Petal. (2017). Enriching Word Vectors with Subword Information.<br>
4. Open Source Tools: TF-IDF, VADER, Scikit-learn<br>
5. Dataset: Emoji Sentiment Ranking 


























