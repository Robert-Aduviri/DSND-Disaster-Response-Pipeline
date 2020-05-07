import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
import joblib
from pandarallel import pandarallel
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.base import ClassifierMixin

from text_preprocessing import tokenize_sentence


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler()) # uncomment to print to stdout
# pandarallel.initialize()


def load_data(database_filepath: str) \
    -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:

    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    # get table name
    table_name = engine.table_names()[0] 
    df = pd.read_sql_table(table_name, engine)
    X = df.message.copy()
    Y = df.iloc[:,5:].copy()
    return X, Y, list(Y.columns)


def build_model() -> Pipeline:

    # TODO: log intermediate steps of the pipeline through wrappers
    tfidf_params = {
        'tokenizer': tokenize_sentence,
        'ngram_range': (1, 2),
        'max_df': 0.9,
        'min_df': 10,
        'max_features': 800,
    }
    clf_params = {
        'RandomForest': {
            'n_estimators': 200,
            'max_depth': 30,
            'n_jobs': -1,
            'random_state': 0
        }
    }
    pipeline = Pipeline([
        ('text_features', Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params))
        ])),
        ('classifier', MultiOutputClassifier(
            RandomForestClassifier(**clf_params['RandomForest'])
        ))
    ])

    gridsearch_params = {
        'text_features__tfidf__max_features': [400, 800],
        # 'text_features__tfidf__ngram_range': [(1, 1), (1, 2)],
        # 'classifier__estimator___n_estimators': [80, 200, 400],
        'classifier__estimator__max_depth': [6, 30]
    }
    cv = GridSearchCV(pipeline, param_grid=gridsearch_params,
                      n_jobs=-1, verbose=2, cv=5)

    return cv


def evaluate_model(
    model: ClassifierMixin, 
    X_test: np.array, Y_test: np.array, 
    category_names: List[str]):

    # TODO: log results
    y_pred = model.predict(X_test)
    report = classification_report(Y_test, y_pred, 
        target_names=category_names)
    print('Labels: ', category_names)
    print('Classification report:')
    print(report)


def save_model(model: ClassifierMixin, model_filepath: str):
    # TODO: output to a pickle file (it does)
    joblib.dump(model, model_filepath)
    # then joblib.load(model_filepath)


def run_model_pipeline(database_filepath: str, model_filepath: str,
                       sample: bool = False):
    logger.info('Loading data...    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    logger.info(f'{len(X)} data points read, with {len(category_names)}'
                ' targets')
    if sample:
        sample_size = 1000
        np.random.seed(0)
        idxs = np.random.choice(range(len(X)), sample_size)
        X, Y = X.iloc[idxs], Y.iloc[idxs]        
        logger.info(f'Sample applied, now only {sample_size} data points')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    logger.info('Building model...')
    model = build_model()
    
    logger.info('Training model...')
    model.fit(X_train, Y_train)
    
    logger.info('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    logger.info(f'Saving model...    MODEL: {model_filepath}')
    save_model(model, model_filepath)

    logger.info('Trained model saved!')

