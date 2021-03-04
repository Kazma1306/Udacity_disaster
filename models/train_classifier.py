import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV
nltk.download(['punkt', 'stopwords', 'wordnet'])

def load_data(database_filepath):
    """
    Loads data from database filepath
    
    Input:
        :database_filepath: path to database to connect
    Returns:
        X: Dataframe, data for training and testing
        Y: Dataframe, labels for data
        category_names: Names  Y
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('select * from messages', engine)
    X = df['message']
    y = df.iloc[:, 4:39]
    category_names = y.columns.tolist()
    return X, y, category_names

def tokenize(text):
    """
    Text preprocess
    Normalizes, tokenizes and lemms text
    
    :Input:
        :text: String, tweet 
    :Returns:
        :clean_tokens: List of strings  
    """
    
    # Normalize
    text = text.lower()
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # Lemmatise words
    cleaned_tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]

    return cleaned_tokens


def build_model():
    """
    Input:
       :None: 
   :Returns:
       :pipeline: Machine Learning pipeline 
    """
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])
    
    parameters = {
            'vect__ngram_range': (1, 2),
            'vect__max_df': (0.5, 7.0),
            'tfidf__use_idf': (True, False)
            
        }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Input:
       model:trained Model 
       X_test: Dataframe
       Y_test: Dataframe, actual labels 
       category_names: List of strings
        
    """
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=category_names)
    for col in category_names:
        
        print(category_names)
        print(classification_report(Y_test[col], Y_pred_df[col]))
        print('--------------------------------------------------------')


def save_model(model, model_filepath):
    """
    Input:
        model: pipeline/model
       :model_filepath: String, filepath where model will be saved
    """
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