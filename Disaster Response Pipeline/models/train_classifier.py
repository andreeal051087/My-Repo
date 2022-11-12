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
    
    ''' 
    load_data
    Function for loading data from the SQLITE database TABLE.
    
    Input:
    database_filepath : database filepath
    
    Returns:
    X, Y, category_names: X -message data, Y -labels/values for the categories, category_names names of the categories.
    
    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM DIS_RESP", engine)
    Y = df.iloc[:,4:]
    X = df['message']
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    
    ''' 
    tokenize
    Function for applyting tokenization to each of our messaes, by splitting into words, reducing to the root form, lower capitalization, and trimming white spaces
    
    Input:
    text : messages text
    
    Returns:
    clean_tokens : list of words per each message: trimmed, lemmatized, lower capitalized
    
    '''    
    
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
    
    ''' 
    build_model
    Function for creating an ML model to use in making predicitions about the message categories, by creating a Pipeline flow, and optimizing it by using a parameter tuning method.
    
    Input:None
    
    Objects description:
    Pipeline:
    CountVectorizer : the count vectoriser transfomer necessary for the data transformation of the corpus into a bag of  words.
    TfidfTransformer: the tfidf transfomer necessary for determining the term frequency and inverse document frequency of the corpus
    MultiOutputClassifier(DecisionTreeClassifier): multi-output classifier used due to plury-dimensionality of data.
    DecisionTreeClassifier: Predictor that will be used for training.
    
    Hyperparameters Tuning:
    GridSearchCV: method used for hyperparameter tuning.
    Parameters: parameters defined for GridSearchCV Hyperparameters Tuning - finding best hyperparameters for optimal results of the model
    
    Returns:
    model: best model built based on the hyperparameter tuning by using GridSearchCV method on our Pipeline.
    
    '''    
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier(random_state=0)))
    ])
    
    parameters = {
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__max_features': ['sqrt', 'log2'],
        'clf__estimator__min_samples_split': [25],
        'clf__estimator__splitter': ['best', 'random']
        
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model
     

def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    evaluate_model
    Function for evaluating results of our trained model, by creating predictions.
    
    Input:
    model: our trained model. 
    X_test: test data containing the input variables/messages.
    Y_test: test data containing the labels/category values. 
    category_names: category names.
    
    Output:
    Y_pred: predictions made by our trained model.
    
    '''
    
    # predict on test data
    Y_pred = model.predict(X_test)
    return 
     

def save_model(model, model_filepath):
    
    '''
    save_model
    Function for saving our trained and tested model in a .pkl file.
    
    Input:
    model: our trained and tested model.
    model_filepath: filepath to the .pkl file containing our trained and tested model.
    
    Output:
    .pkl file: file saved in the model filepath containing our trained and tested model.
    
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    
    '''
    main function: starting point for the execution of the process by triggering the loading data and splitting in test and train sets, training, testing, and evaluation of the model
    
    Input: none
    Output: Trained model saved in a pickle file
    
    '''
    
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