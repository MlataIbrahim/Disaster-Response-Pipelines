# coding: utf-8
# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# - Import Python libraries
# - Load dataset from database
# - Define feature and target variables X and Y

# import libraries

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# load data from database
engine = create_engine('sqlite:///Disatser.db')
df = pd.read_sql_table('messages', engine)
X = df['message']
Y = df.loc[:,'related':]
Y = Y.astype('int')
# ### 2. Write a tokenization function to process your text data
def tokenize(text):
    # (1) Normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ",text.lower())
    # (2) Tokenization
    words = word_tokenize(text)
    # (3) Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    # (4) Lemmatization -- Reduce words to their root form
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    # (5) Stemming -- Reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in words]

    return stemmed
# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.
pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(
                            OneVsRestClassifier(LinearSVC())))])
# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=45)
# train pipeline
pipeline.fit(X_train,y_train)
# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.
# predict on test data
# report the f1 score, precision and recall
def report(model, X_test, y_test):
    '''
    Function to generate classification report
    Input: X_test & y_test
    Output: Prints the Classification report
    '''
    y_pred = model.predict(X_test)
    for i, col in enumerate(y_test):
        print(col)
        print(classification_report(y_test[col], y_pred[:, i]))
report(pipeline, X_test, y_test)
# ### 6. Improve your model
# Use grid search to find better parameters.
parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }
cv = GridSearchCV(pipeline, param_grid=parameters)
# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!
cv.fit(X_train, y_train)
report(cv, X_test, y_test)
# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF
pipeline_1 = Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', DecisionTreeClassifier())
                     ])
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=45)
# train pipeline
clf_1 = pipeline_1.fit(X_train,y_train)
# test the model
y_pred_test_1 = pipeline_1.predict(X_test)
# report after improving
report(pipeline_1, y_test, y_pred_test_1)
# Exporting the model
model = pickle.dumps('classifier')
