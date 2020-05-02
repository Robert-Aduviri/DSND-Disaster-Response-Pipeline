import sys
from typing import Tuple, List
import pandas as pd
from pandarallel import pandarallel
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from text_preprocessing import preprocess_sentence

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


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X['message'] = X.message.parallel_apply(preprocess_sentence)
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