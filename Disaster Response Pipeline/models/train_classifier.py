import sys
# import libraries

import numpy as np
import pandas as pd
import pickle

from sqlalchemy import create_engine
import sqlite3

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM DIS_RESP", engine)
    Y = df.iloc[:,4:]
    X = df['message']
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    # we split into words
    words = word_tokenize(text)
    # we lemmatize (we reduce to root form)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for w in words:
        clean_tok = lemmatizer.lemmatize(w)
        # changing all to lower capital leters
        clean_tok_lower = clean_tok.lower()
         # trimming the blank spaces before and after
        clean_tok_strip = clean_tok_lower.strip()
         # adding elements to the list
        clean_tokens.append(clean_tok_strip)
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier(random_state=0)))
    ])
    
    parameters = {
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__max_features': ['sqrt', 'log2'],
        'clf__estimator__min_samples_split':[20]
        
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model
     

def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    Y_pred = model.predict(X_test)
    return 
     

def save_model(model, model_filepath):
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
        model.fit(X_train, Y_train)
        model=model.best_estimator_
        
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