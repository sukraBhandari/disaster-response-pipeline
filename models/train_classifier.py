import sys
import re
import nltk

import pandas as pd
from sqlalchemy import create_engine

from sklearn.externals import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfTransformer)
from sklearn.metrics import classification_report


from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    '''
    Reads the sqlite database file into pandas dataframe
    Defines feature and target variables X and Y
    
    Parameters
    ----------
    database_filepath : database file path
    
    Returns:
    X : Feature variables
    Y : Target variables
    category_names : Target variables names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM disaster_messages_clean", engine)
    
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

def tokenize(text):
    '''
    Tokensizes the text
    
    Parameters
    ----------
    text : text to be tokenized
    
    Returns:
    --------
    tokens : list of token
    '''
    # instantiate Lemmatizer and Stemmer
    stemmer = PorterStemmer()    
    lemmatizer = WordNetLemmatizer()
    
    # get set of english stopwords
    stop_words_nltk = set(stopwords.words('english'))
    
    # remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize the text
    tokens = word_tokenize(text)
    
    # remove stopwords from tokens list
    tokens = [word for word in tokens if not word in stop_words_nltk]
    
    # reduce words to their stems
    tokens = [stemmer.stem(word) for word in tokens]
    
    # reduce words to their root form
    tokens = [lemmatizer.lemmatize(word, 'n') for word in tokens]
    
    return tokens


def build_model():
    '''
    Builds a pipeline for dat manipuation and transformation
    
    Parameters
    ----------
    None
    
    Returns
    -------
    model : classifier
    '''
   
    parameters = {
                'clf__estimator__n_estimators': [50, 100, 300],
                'clf__estimator__learning_rate': [0.5, 1.0, 2.0],
                'vect__max_df': [0.9, 1.0],
                'vect__ngram_range': [(1, 1), (1, 2)],
                'tfidf__use_idf': [True, False]
                }
    ada_clf = AdaBoostClassifier()
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(ada_clf))
                        ])
    model = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=10)
    return model
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the model and prints classification report
    
    Parameters
    ----------
    model : classification model
    X_test : test features
    Y_test : test target values
    category_names : target category names
    
    Returns
    -------
    None
    '''
    # model prediction on test dataset
    y_pred = model.predict(X_test)
    
    # print classification report for each category
    for i, column in enumerate(category_names):
        print("-------------", column, "--------------")
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))
    

def save_model(model, model_filepath):
    '''
    Exports model as a pickle file 
    '''
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Best Estimator...')
        print(model.best_estimator_)
        
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