import sys, logging, argparse
from model_pipeline import run_model_pipeline

def main():
    parser = argparse.ArgumentParser(
        'Please provide the filepath of the disaster messages database '\
        'as the first argument and the filepath of the pickle file to '\
        'save the model to as the second argument. \n\nExample: python '\
        'train_classifier.py ../data/DisasterResponse.db classifier.pkl'
    )
    parser.add_argument('database_filepath', type=str, 
        help='filepath of the disaster messages database (.db)')
    parser.add_argument('model_filepath', type=str, 
        help='filepath of the file to save the model to (.pkl)')
    parser.add_argument('-s', '--sample', action='store_true', 
        help='run with only a sample of the data (1000 rows)')

    args = parser.parse_args()
    run_model_pipeline(args.database_filepath, args.model_filepath, 
                       args.sample)


if __name__ == '__main__':
    import logging

    logging.basicConfig(
        filename='data/model.log',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        datefmt='%d-%b-%y %H:%M:%S',
        level=logging.INFO
    )
    main()