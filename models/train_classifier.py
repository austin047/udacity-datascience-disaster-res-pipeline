import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import pickle


sys.path.append("./models")
from starting_verb_extractor import StartingVerbExtractor

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    '''
    Description: Load data from database given sqlite database given 'database_filepath' then return feature, target values and target value column names
    
    Arguments:
        database_filepath: File path to database in this case sqlite database 
        
    Return:
        X: Feature
        Y: Target values
        category_names: Targeted values category names
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    # Read from database table
    df = pd.read_sql_table('message_and_category', engine)  
    
    # Drop child_alone category that has only zeros. Otherwise SGDClassifier will not be able to run.
    df = df.drop(['child_alone'], axis=1)
    
    # Exact features, target values and column names
    X = df.message
    Y = df[df.columns[4:]]
    category_names = df[df.columns[4:]].columns
    
    return X, Y, category_names


def tokenizer(text):
    '''
    Description: A tokenization function to process the text data other function of this function include 
        - Replace urls
        - tokenize words
        - lemmatization
        - Normalization and strip trailing white spaces 
    
    Arguments:
        text: A raw text to be tokenized
        
    Return:
        None
    '''
        
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Get all url's in text
    detected_urls = re.findall(url_regex, text)
    
    # replace all urls in text with 'urlplaceholder'
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Tokenize 'text' and return tokenized array
    tokens = word_tokenize(text)
    
    # Initialize Lemmatizer, to reduce all words to thier normal form
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    
    # Go through token to to lammatize, normalize and strip all white spaces
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens
    
def build_model():
    '''
    Description: Build the machine learning model, use sklearn Pipeline to create a machine learning pipeline of estimators
    
    Arguments:
        database_filepath:
        
    Return 
    '''
    pipeline  = Pipeline([
        ('features', FeatureUnion([ # 

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenizer)), # Generate bag of words bag given document
                ('tfidf', TfidfTransformer()) # Gives weighting to words matrix from previous estimator
            ])),

            ('starting_verb', StartingVerbExtractor(tokenizer=tokenizer)) #Extract the starting verbs of a sentence 
        ])),

        ('clf', MultiOutputClassifier(SGDClassifier(random_state=42)))
    ])
    
    parameters = {
    'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)

    return cv
    
#     return  pipeline


def evaluate_model(model, X_test, Y_test, category_names, has_params=False):
    '''
    Description: Evalute the model by runing prediction and comparing with actual targeted values 
    
    Arguments:
        model: Machine learning model
        X_test: Test data set to test the model
        Y_test: Tageted value values 
        category_names; Category names of the tagrgeted values
        has_params: Boolean to determing if to display the best scores or Parameter
        
    Return 
        None
    '''    
    Y_pred = model.predict(X_test)



    
    if has_params: #If params are used display best params an best scores
        print("\nBest Parameters:", model.best_params_)
        print("\nBest Scores:",model.best_score_)
    
    accuracy = (Y_pred == Y_test).mean().mean()
    print("Overall Accuracy For Model:", accuracy)

    for i, column in enumerate(Y_test.columns):
        print('Model Performance with Category: {}'.format(column))
        print('Accuracy for {}: {}'.format(column, accuracy_score(Y_test[column],  Y_pred.T[i][:])))
        print(classification_report(np.array(Y_test[column]),Y_pred.T[i])) # Transpose y to 
        
#     for i, col in enumerate(columns): #Go through colouns 
#         print(i, col)
#         print(classification_report(y_test.to_numpy()[:1],  y_pred[:1], target_names=columns))
#         print(accuracy_score(y_test.to_numpy()[:, 1],  y_pred[:, 1]))


def save_model(model, model_filepath):
    '''
    Description: Save the machine learning model as a model as a pickle file
    
    Arguments:
        model: Machine learning model
        model_filepath: filepath to pickle file '.pkl'
        
    Return 
    '''
    pickle_filepath = 'model_filepath'

    pickle.dump(model, open(pickle_filepath, 'wb'))


    

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
    
    