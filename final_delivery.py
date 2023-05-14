# Import basic libraries
import os
import numpy as np
import pandas as pd

# Import libraries for text processing
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder 

# Import libraries for machine learning
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Import libraries for logging
import logging
from tqdm import tqdm

# Load product and review data for CDs and vinyls from the training set
data_dir = ''
categories = ['CDs_and_Vinyl', 'Grocery_and_Gourmet_Food', 'Toys_and_Games']

file_path = os.path.join(data_dir, categories[0], 'test2', 'product_test.json')
product_test = pd.read_json(file_path)

file_path = os.path.join(data_dir, categories[0], 'test2', 'review_test.json')
review_test = pd.read_json(file_path)

# Merge product and review data
test_data = review_test.merge(product_test, on='asin', how='left')

# Fill in any missing values
test_data['reviewText'].fillna('', inplace=True)
test_data['summary'].fillna('', inplace=True)

# Give each review a unique ID
test_data['reviewID'] = test_data.index

# Run sentiment analysis on the review text and summary
# Columns: neg, neu, pos, compound

sid = SentimentIntensityAnalyzer()
review_sentiments = pd.DataFrame(columns=['reviewID', 'reviewText_neg', 'reviewText_neu', 'reviewText_pos', 'reviewText_compound', 'summary_neg', 'summary_neu', 'summary_pos', 'summary_compound'])

for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc="Sentiment Analysis"):
    review_text_sentiment = sid.polarity_scores(row['reviewText'])
    summary_text_sentiment = sid.polarity_scores(row['summary'])
    
    sentiment_row = {'reviewID': row['reviewID'],
                     'reviewText_neg': review_text_sentiment['neg'],
                     'reviewText_neu': review_text_sentiment['neu'],
                     'reviewText_pos': review_text_sentiment['pos'],
                     'reviewText_compound': review_text_sentiment['compound'],
                     'summary_neg': summary_text_sentiment['neg'],
                     'summary_neu': summary_text_sentiment['neu'],
                     'summary_pos': summary_text_sentiment['pos'],
                     'summary_compound': summary_text_sentiment['compound']}
    review_sentiments = pd.concat([review_sentiments, pd.DataFrame([sentiment_row])], ignore_index=True)
    # review_sentiments = pd.concat(sentiment_row, ignore_index=True)

# Save the sentiment data to a csv file for future use
file_path = os.path.join(data_dir, 'review_sentiments_test_2.csv')
review_sentiments.to_csv(file_path, index=False)
review_sentiments = pd.read_csv('review_sentiments_test_2.csv')

# test_data = test_data.merge(review_sentiments, on='reviewID', how='left')

# # Process the columns that are not numeric
# encoder = LabelEncoder()

# # Define a function to extract the format information from the "style" column
# def extract_format(style):
#     if style is None:
#         return "None"
#     else:
#         return style.get("Format:", "None").strip()

# # Apply the function to the "style" column to extract the format information
# test_data["style"] = test_data["style"].apply(extract_format)

# # Encode the columns
# encoder = LabelEncoder()
# test_data["style"] = encoder.fit_transform(test_data["style"])

# # Encode the "verified" column
# test_data["verified"] = encoder.fit_transform(test_data["verified"])

# # Encode the "reviewerID" column
# test_data["reviewerID"] = encoder.fit_transform(test_data["reviewerID"])

# # Encode the "vote"" column
# test_data["vote"] = test_data["vote"].apply(lambda x: float(x.replace(",", "")) if x is not None else 0)

# # Encode the "image" column
# test_data["image"] = test_data["image"].apply(lambda x: len(x) if x is not None else 0)

# # Compute the length of reviewText and summary columns
# test_data["reviewText_len"] = test_data["reviewText"].apply(len)
# test_data["summary_len"] = test_data["summary"].apply(len)

# # # Filter out the reviews that are not verified, have no votes, and have no images unless there is no verified and voted reviews
# # training_data = training_data[(training_data["verified"] == 1) | (training_data["vote"] > 0) | (training_data["image"] > 0)]

# # Normalize the compound scores
# test_data["reviewText_compound_norm"] = (test_data["reviewText_compound"] - test_data["reviewText_compound"].mean()) / test_data["reviewText_compound"].std()
# test_data["summary_compound_norm"] = (test_data["summary_compound"] - test_data["summary_compound"].mean()) / test_data["summary_compound"].std()

# # Calculate the absolute difference between the normalized compound scores and the awesomeness
# test_data["reviewText_compound_diff"] = abs(test_data["reviewText_compound_norm"] - test_data["awesomeness"])
# test_data["summary_compound_diff"] = abs(test_data["summary_compound_norm"] - test_data["awesomeness"])

# # Calculate the average difference between the normalized compound scores and the awesomeness for each asin
# compound_diff_mean = test_data.groupby("asin")[["reviewText_compound_diff", "summary_compound_diff"]].mean()

# # Sort the reviews for each asin by the average difference between the normalized compound scores and the awesomeness
# compound_diff_mean["compound_diff_mean"] = compound_diff_mean.mean(axis=1)
# compound_diff_mean = compound_diff_mean.sort_values("compound_diff_mean", ascending=False)

# # Keep the top 2/3 of the reviews for each asin
# num_asins = len(compound_diff_mean)
# top_reviews_per_asin = int(num_asins * 2/3)
# top_asins = compound_diff_mean.iloc[:top_reviews_per_asin].index
# test_data = test_data[test_data["asin"].isin(top_asins)]

# # Aggregate the training data by asin
# test_data = test_data.groupby("asin").agg({
#     "reviewerID": "count",
#     "unixReviewTime": ["min", "max", "mean", "std"],
#     "verified": ["mean", "sum"],
#     "vote": ["mean", "sum"],
#     "image": ["mean", "sum"],
#     "style": ["mean", "sum"],
#     "reviewText_neg": ["mean", "std"],
#     "reviewText_neu": ["mean", "std"],
#     "reviewText_pos": ["mean", "std"],
#     "reviewText_compound": ["mean", "std"],
#     "summary_neg": ["mean", "std"],
#     "summary_neu": ["mean", "std"],
#     "summary_pos": ["mean", "std"],
#     "summary_compound": ["mean", "std"],
#     "reviewText_len": ["mean", "std"],
#     "summary_len": ["mean", "std"],
# }).reset_index()

# test_data["reviewText_compound"] += 1
# test_data["summary_compound"] += 1
# # Replace NaN values with 0
# test_data.fillna(0, inplace=True)


