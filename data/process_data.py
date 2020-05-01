import sys, os, logging
from typing import Tuple
import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

def extract(messages_filepath: str, categories_filepath: str) \
    -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the messages and categories csv files into dataframes"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    logger.info(f'{len(messages)} records found in messages')
    logger.info(f'{len(messages)} records found in categories')
    return messages, categories


def transform_messages(messages: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates from messages dataframe"""
    messages = messages.drop_duplicates('id', keep='last')
    return messages


def transform_categories(categories: pd.DataFrame) -> pd.DataFrame:
    """Split and parse categories to a numeric format, 
    also removing duplicates
    """
    # split the column to get the actual categories, while keeping the 
    # message id
    categories = categories.set_index('id').categories.str \
                           .split(';', expand=True)
    # the first row is used to extract the column names
    row = categories.iloc[0]
    categories.columns = [x.split('-')[0] for x in row]
    for column in categories:
        # the values are in the last character of the string
        categories[column] = categories[column].str.split('-').str.get(-1)
        categories[column] = categories[column].astype(int)
    # get back the message id as a column
    categories = categories.reset_index()
    # NOTE: there are duplicates (based on id) which have slightly different 
    #       categories. The latest one is taken
    categories = categories.drop_duplicates('id', keep='last')
    return categories


def transform(messages: pd.DataFrame, categories: pd.DataFrame) \
    -> pd.DataFrame:
    """Transforms and merges the messages and categories dataframes"""
    logger.info('Cleaning data...')
    messages = transform_messages(messages)
    categories = transform_categories(categories)
    data = pd.merge(messages, categories, on='id')
    logger.info(f'Joined {len(messages)} unique messages')
    return data


def load(data: pd.DataFrame, db_filepath: str, table_name: str):
    """Save data in a table inside a SQLite db 

    Args:
    data: dataframe
    db_filepath: SQLite database file name
    table_name: table name inside the SQLite database
    """
    if os.path.exists(db_filepath):
        os.remove(db_filepath)
    
    logger.info(f'Saving data...    DATABASE: {db_filepath}')
    engine = create_engine(f'sqlite:///{db_filepath}')
    data.to_sql(table_name, engine, index=False)


def etl(messages_filepath: str = 'data/messages.csv', 
        categories_filepath: str = 'data/categories.csv',
        db_filepath: str = 'data/DisasterResponse_preprocessed.db') \
        -> pd.DataFrame: 
    """Extract, Transform and Load preprocessed data from the csv files

    Args:
    messages_file: messages csv file name
    categories_file: categories csv file name
    db_filepath: output data SQLite database file name

    Returns:
    data: preprocessed data
    """
    logger.info(f'Loading data into {db_filepath}...')
    logger.info(f'MESSAGES: {messages_filepath} | CATEGORIES: {categories_filepath}')
    messages, categories = extract(messages_filepath, categories_filepath)
    data = transform(messages, categories)
    load(data, db_filepath, 'message')
    return data
    

def main():
    if len(sys.argv) == 4:
        logger.info('Starting ETL...')
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        etl(messages_filepath, categories_filepath, database_filepath)
        logger.info('ETL process finished successfully')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    import logging

    logging.basicConfig(
        filename='data/etl.log',
        format='%(asctime)s - %(levelname)s - %(message)s', 
        datefmt='%d-%b-%y %H:%M:%S',
        level=logging.INFO
    )
    main()