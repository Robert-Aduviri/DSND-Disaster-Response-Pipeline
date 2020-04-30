import os
from process_data import extract, transform, load, etl

def test_etl():
    messages_filename = 'data/messages.csv'
    categories_filename = 'data/categories.csv'
    db_filename = 'data/DisasterResponse_preprocessed.db'
    if os.path.exists(db_filename):
        os.remove(db_filename)
    data = etl(messages_filename, categories_filename, db_filename)
    os.remove(db_filename)
    assert data.shape == (26180, 40)