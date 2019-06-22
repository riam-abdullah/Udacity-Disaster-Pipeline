import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

# import libraries
import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """
    load data from database
    param:database file path
    return: X,y, category names list
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterMessage', con=engine)
    X = df.loc[:, 'message']
    y = df.iloc[:, 4:]
    category_names = y.columns.tolist()
    
    return X, y, category_names


def tokenize(text):
    """
    Will tokenize and lemmatize the message text
    param: text
    return: list of clean tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds the model pipeline
    param:None
    returns:grid serach that contain pipeline
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])
    parameters = {'clf__estimator__n_estimators': [50, 30],
              'clf__estimator__min_samples_split': [3, 2] 
    }

    cv = GridSearchCV(pipeline, param_grid= parameters, verbose=2)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluates model's performance on the test set.
    param : model
    param : X_test
    param : y_test
    param : category_names
    return : None
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, y_test, target_names = category_names))
    

def save_model(model, model_filepath):
    '''
    Save model in pickle format
    param: model
    param: model_filepath
    return: None
    '''
    pickle.dump(model, open(model_filepath, "wb"))


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