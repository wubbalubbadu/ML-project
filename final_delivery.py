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
product_training = pd.read_json(file_path)

file_path = os.path.join(data_dir, categories[0], 'test2', 'review_test.json')
review_training = pd.read_json(file_path)

# Merge product and review data
training_data = review_training.merge(product_training, on='asin', how='left')
# Fill in any missing values
training_data['reviewText'].fillna('', inplace=True)
training_data['summary'].fillna('', inplace=True)

# Give each review a unique ID
training_data['reviewID'] = training_data.index

# Run sentiment analysis on the review text and summary
# Columns: neg, neu, pos, compound

# sid = SentimentIntensityAnalyzer()

# review_sentiments = pd.DataFrame(columns=['reviewID', 'reviewText_neg', 'reviewText_neu', 'reviewText_pos', 'reviewText_compound', 'summary_neg', 'summary_neu', 'summary_pos', 'summary_compound'])

# for index, row in tqdm(training_data.iterrows(), total=training_data.shape[0], desc="Sentiment Analysis"):
#     review_text_sentiment = sid.polarity_scores(row['reviewText'])
#     summary_text_sentiment = sid.polarity_scores(row['summary'])
    
#     sentiment_row = {'reviewID': row['reviewID'],
#                      'reviewText_neg': review_text_sentiment['neg'],
#                      'reviewText_neu': review_text_sentiment['neu'],
#                      'reviewText_pos': review_text_sentiment['pos'],
#                      'reviewText_compound': review_text_sentiment['compound'],
#                      'summary_neg': summary_text_sentiment['neg'],
#                      'summary_neu': summary_text_sentiment['neu'],
#                      'summary_pos': summary_text_sentiment['pos'],
#                      'summary_compound': summary_text_sentiment['compound']}
    
#     review_sentiments = review_sentiments.append(sentiment_row, ignore_index=True)

# # Save the sentiment data to a csv file for future use
# file_path = os.path.join(data_dir, categories[0], 'csv', 'review_sentiments.csv')
# review_sentiments.to_csv(file_path, index=False)
review_sentiments = pd.read_csv('review_sentiments_test_2.csv')
training_data = training_data.merge(review_sentiments, on='reviewID', how='left')

# Process the columns that are not numeric
encoder = LabelEncoder()

# Define a function to extract the format information from the "style" column
def extract_format(style):
    if style is None:
        return "None"
    else:
        return style.get("Format:", "None").strip()

# Apply the function to the "style" column to extract the format information
training_data["style"] = training_data["style"].apply(extract_format)

# Encode the columns
encoder = LabelEncoder()
training_data["style"] = encoder.fit_transform(training_data["style"])

# Encode the "verified" column
training_data["verified"] = encoder.fit_transform(training_data["verified"])

# Encode the "reviewerID" column
training_data["reviewerID"] = encoder.fit_transform(training_data["reviewerID"])

# Encode the "vote"" column
training_data["vote"] = training_data["vote"].apply(lambda x: float(x.replace(",", "")) if x is not None else 0)

# Encode the "image" column
training_data["image"] = training_data["image"].apply(lambda x: len(x) if x is not None else 0)

# Compute the length of reviewText and summary columns
training_data["reviewText_len"] = training_data["reviewText"].apply(len)
training_data["summary_len"] = training_data["summary"].apply(len)

# # Filter out the reviews that are not verified, have no votes, and have no images unless there is no verified and voted reviews
# training_data = training_data[(training_data["verified"] == 1) | (training_data["vote"] > 0) | (training_data["image"] > 0)]

# Normalize the compound scores
print(training_data.columns)
training_data["reviewText_compound_norm"] = (training_data["reviewText_compound"] - training_data["reviewText_compound"].mean()) / training_data["reviewText_compound"].std()
training_data["summary_compound_norm"] = (training_data["summary_compound"] - training_data["summary_compound"].mean()) / training_data["summary_compound"].std()

# Calculate the absolute difference between the normalized compound scores and the awesomeness
training_data["reviewText_compound_diff"] = abs(training_data["reviewText_compound_norm"] - training_data["awesomeness"])
training_data["summary_compound_diff"] = abs(training_data["summary_compound_norm"] - training_data["awesomeness"])

# Calculate the average difference between the normalized compound scores and the awesomeness for each asin
compound_diff_mean = training_data.groupby("asin")[["reviewText_compound_diff", "summary_compound_diff"]].mean()

# Sort the reviews for each asin by the average difference between the normalized compound scores and the awesomeness
compound_diff_mean["compound_diff_mean"] = compound_diff_mean.mean(axis=1)
compound_diff_mean = compound_diff_mean.sort_values("compound_diff_mean", ascending=False)

# Keep the top 2/3 of the reviews for each asin
num_asins = len(compound_diff_mean)
top_reviews_per_asin = int(num_asins * 2/3)
top_asins = compound_diff_mean.iloc[:top_reviews_per_asin].index
training_data = training_data[training_data["asin"].isin(top_asins)]
training_data

# Aggregate the training data by asin
training_data = training_data.groupby("asin").agg({
    "reviewerID": "count",
    "unixReviewTime": ["min", "max", "mean", "std"],
    "verified": ["mean", "sum"],
    "vote": ["mean", "sum"],
    "image": ["mean", "sum"],
    "style": ["mean", "sum"],
    "reviewText_neg": ["mean", "std"],
    "reviewText_neu": ["mean", "std"],
    "reviewText_pos": ["mean", "std"],
    "reviewText_compound": ["mean", "std"],
    "summary_neg": ["mean", "std"],
    "summary_neu": ["mean", "std"],
    "summary_pos": ["mean", "std"],
    "summary_compound": ["mean", "std"],
    "reviewText_len": ["mean", "std"],
    "summary_len": ["mean", "std"],
}).reset_index()

# Add +1 to compound columns to avoid negative values
training_data["reviewText_compound"] += 1
training_data["summary_compound"] += 1
# Replace NaN values with 0
training_data.fillna(0, inplace=True)

column_dict = {
    #"reviewerID": ["count"],
    "unixReviewTime": ["min", "max", "mean", "std"],
    "verified": ["mean", "sum"],
    "vote": ["mean", "sum"],
    "image": ["mean", "sum"],
    "style": ["mean", "sum"],
    "reviewText_neg": ["mean", "std"],
    "reviewText_neu": ["mean", "std"],
    "reviewText_pos": ["mean", "std"],
    "reviewText_compound": ["mean", "std"],
    "summary_neg": ["mean", "std"],
    "summary_neu": ["mean", "std"],
    "summary_pos": ["mean", "std"],
    "summary_compound": ["mean", "std"],
    "reviewText_len": ["mean", "std"],
    "summary_len": ["mean", "std"],
}
column_list = []
for k in column_dict:
    for n in column_dict[k]:
        column_list.append((k, n))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Define the columns to normalize
#cols_to_normalize = ['unixReviewTime', 'verified', 'vote', 'image', 'style', 'reviewText_neg', 'reviewText_neu', 'reviewText_pos', 'reviewText_compound', 'summary_neg', 'summary_neu', 'summary_pos', 'summary_compound', 'reviewText_len', 'summary_len']
#cols_to_normalize = [('verified','mean')]
# Normalize the data using the MinMaxScaler
#scaled_df = scaler.fit_transform(training_data[cols_to_normalize])
# training_data
training_data[column_list] = scaler.fit_transform(training_data[column_list])
# training_data

# Merge the training data with the awesomeness data
file_path = os.path.join(data_dir, categories[0], 'train', 'product_training.json')
product_training = pd.read_json(file_path)
training_data.columns = training_data.columns.to_flat_index()

training_data['asin'] = training_data[('asin', '')]

training_data = training_data.merge(product_training, on='asin', how='left')

# Visualize the absolute correlation between the features on "awesomeness"
#training_data.corr()["awesomeness"].abs().sort_values(ascending=False)
training_data

training_data = training_data.drop(training_data.columns[1], axis=1)

# Prepare the data for training
# Keep only the most important features for predicting awesomeness
X_test_new = training_data[[    
    ('reviewText_pos', 'mean'),    
    ('summary_neu', 'mean'),
    ('reviewText_neg', 'mean'),
    ('summary_neg', 'std'),
    ('summary_neg', 'mean'),
    ('reviewText_neu', 'mean'),
    ('reviewText_neu', 'std')
]].values

# Train the model on naive bayes, decision tree, and random forest classifiers
best_model = joblib.load('final_model.pkl')
predictions = best_model.predict(X_test_new)

predictions
