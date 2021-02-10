'''
    File name: process_data.py
    Author: Andres Hoyos
    Date created: 09/Feb/2021
    Date last modified: 09/Feb/2021
    Python Version: 3.6
'''
#import libraries

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


import numpy as np 
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import sys
import os
import re
import string
from sqlalchemy import create_engine
import pickle

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin



def load_data(database_filepath):
       
    '''
    load_data from a database and extract variables for modelling
    
    Input:
    database_filepath: Path of database containing clean table 
       
    output: 
    X: DataFrame of features
    y: Dataframe of targets
    
    '''  
    engine = create_engine("sqlite:///" + database_filepath)
    table_name = "T_Categorized_Messages1"
    df = pd.read_sql_table(table_name, engine)

    # Extract X and y variables from the data for modelling
    X = df["message"]
    y = df.iloc[:, 4:]

    return X, y

def tokenize(text):
    
    '''
    Tokenize a text to make it suitable for modelling
    
    Input:
    text: string with the test to tokenize 
       
    output: 
    lemmed: list of cleaned tokens
    '''
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 

    # Extract the word tokens from the provided text
    words = word_tokenize(text)

    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # Lemmatization
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed


def build_model():
    
    '''
    Define the pipeline for the text data and define the models parameters for the training
    
    Input:
    None 
       
    output: 
    cv: Model definition
    '''
    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text_pipeline",
                            Pipeline(
                                [
                                    (
                                        "count_vectorizer",
                                        CountVectorizer(tokenizer=tokenize),
                                    ),
                                    ("tfidf_transformer", TfidfTransformer()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("classifier", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )
    
    #parameter pro the grid search
    parameters = {
            'classifier__estimator__max_depth': [ 20, 40, 60],
            'classifier__estimator__n_estimators': [5, 10, 15]
    }
    
    #model Definition
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    
    return cv


def evaluate_model(model, X_test, y_test):
    
    '''
    evaluates the model
    
    Input:
    Model: models results
    X_test: DataFrame of test values
    y_test: DataFrame with the labeled targets
       
    output: 
    None
    '''
    #Predictions
    y_prediction_test = model.predict(X_test)
    
    #print reports
    print(classification_report(y_test.values, y_prediction_test, target_names=y_test.columns.values))


def save_model(model, model_filepath):
    
    '''
    saves the model
    
    Input:
    model: fitted model
    model_filepath: path for saving
       
    output: 
    None
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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