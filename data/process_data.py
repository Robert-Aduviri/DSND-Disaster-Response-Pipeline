from typing import Tuple
import pandas as pd
from sqlalchemy import create_engine


def extract(messages_filename: str, categories_filename: str) \
    -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the messages and categories csv files into dataframes"""
    messages = pd.read_csv(messages_filename)
    categories = pd.read_csv(categories_filename)
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
    messages = transform_messages(messages)
    categories = transform_categories(categories)
    data = pd.merge(messages, categories, on='id')
    return data


def load(data: pd.DataFrame, db_filename: str, table_name: str):
    """Save data in a table inside a SQLite db 

    Args:
    data: dataframe
    db_filename: SQLite database file name
    table_name: table name inside the SQLite database
    """
    engine = create_engine(f'sqlite:///{db_filename}')
    data.to_sql(table_name, engine, index=False)


def etl(messages_file: str = 'data/messages.csv', 
        categories_file: str = 'data/categories.csv',
        db_file: str = 'DisasterResponse_preprocessed.db') \
        -> pd.DataFrame: 
    """Extract, Transform and Load preprocessed data from the csv files

    Args:
    messages_file: messages csv file name
    categories_file: categories csv file name
    db_file: output data SQLite database file name

    Returns:
    data: preprocessed data
    """
    messages, categories = extract(messages_file, categories_file)
    data = transform(messages, categories)
    load(data, db_file, 'message')
    return data
    