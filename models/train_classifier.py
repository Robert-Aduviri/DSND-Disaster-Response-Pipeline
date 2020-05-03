import sys
from model_pipeline import run_model_pipeline

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        run_model_pipeline(database_filepath, model_filepath)
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    import logging

    logging.basicConfig(
        filename='data/model.log',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        datefmt='%d-%b-%y %H:%M:%S',
        level=logging.INFO
    )
    main()