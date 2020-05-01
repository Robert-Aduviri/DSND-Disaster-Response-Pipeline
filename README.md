# Disaster Response Pipeline

[Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) project, covering ETL and Machine Learning pipelines.

# Project components and instructions

## 1. ETL Pipeline
Data cleaning pipeline, in process_data.py:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

To run ETL pipeline that cleans data and stores in database
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

## 2. ML Pipeline
Machine Learning pipeline, in train_classifier.py:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

To run ML pipeline that trains classifier and stores it
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

## 3. Flask Web App
We are providing much of the flask web app for you, but feel free to add extra features depending on your knowledge of flask, html, css and javascript. For this part, you'll need to:

- Modify file paths for database and model as needed
- Add data visualizations using Plotly in the web app. One example is provided for you

Run the following command in the app's directory to run your web app.
    `python run.py`

Go to http://0.0.0.0:3001/