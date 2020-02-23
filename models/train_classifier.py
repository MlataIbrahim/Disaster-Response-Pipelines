# imports libraries
import sys
import sys, pickle, re
import pandas as pd
import numpy as np

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine


def load_data(database_filepath):
        """
    Loads data from SQLite Database
    INPUT:
    database_filepath: SQLite database file
    OUPUT:
    X : Features dataframe
    Y : Target dataframe
    category_names list: Target labels
    """
    # connect the database
        engine = create_engine(f"sqlite:///{database_filepath}")
    # fetch the table
        df = pd.read_sql_table('Disaster', engine)
    # select features
        X = df['message']
    # select targets
        Y = df.loc[:,'related':]
    # Y['related'] contains three distinct values mapping extra values to `1`
        Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)
        category_names = Y.columns
        return X,Y,category_names

def tokenize(text):
        """
    Tokenizes text data
    INPUT:
    text str: Messages as text data
    OUPUT:
    words list: Processed text after all text processing steps
    """
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


def build_model():
        """
    Build model with GridSearchCV

    OUPUT:
    Trained model
    """
    # build a pipeline
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    # params dict to tune a model
        parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'tfidf__use_idf': (True, False),
            'clf__estimator__min_samples_split': [2, 4],
            'clf__estimator__max_depth': [25, 100, 200],
        }
    # instantiate a gridsearchcv object with the params defined
        cv = GridSearchCV(pipeline, param_grid=parameters, verbose=4, n_jobs=6)

        return cv

def evaluate_model(model, X_test, Y_test, category_names):
        '''
    Function to evaluate a model and return the classificatio and accurancy score.
    Inputs:
          model: trained model
          X_test: Test features
          Y_test: Test targets
    Outputs: Prints the Classification report & Accuracy Score
    '''
    # predict on test data with tuned params
        y_pred = model.predict(X_test)
    # print classification report
        print(classification_report(Y_test.values, y_pred,
        target_names=category_names))

def save_model(model, model_filepath):
        '''
    Function to save the model
    Input: model and the file path to save the model
    Output: save the model as pickle file in the give filepath
    '''
        pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train.astype(int))

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
