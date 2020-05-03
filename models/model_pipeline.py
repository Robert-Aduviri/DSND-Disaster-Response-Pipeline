import logging
from typing import Tuple, List
import pandas as pd
from pandarallel import pandarallel
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from text_preprocessing import preprocess_sentence


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler()) # uncomment to print to stdout
pandarallel.initialize()


def load_data(database_filepath: str) \
    -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:

    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    # get table name
    table_name = engine.table_names()[0] 
    df = pd.read_sql_table(table_name, engine)
    X = df[['message']].copy()
    Y = df.iloc[:,5:].copy()
    return X, Y, list(Y.columns)


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def run_model_pipeline(database_filepath: str, model_filepath: str):
    logger.info('Loading data...    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X['message'] = X.message.parallel_apply(preprocess_sentence)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    logger.info('Building model...')
    model = build_model()
    
    logger.info('Training model...')
    model.fit(X_train, Y_train)
    
    logger.info('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    logger.info('Saving model...    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    logger.info('Trained model saved!')

