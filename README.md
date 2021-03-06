# Disaster Response Pipeline Project
Data Engineering Section -- Project: Disaster Response Pipeline

## Table of Contents
1. [Project Motivation](#ProjectMotivation)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Results](#results)

## Project Motivation <a name="ProjectMotivation"></a>
This is a Udacity Nanodegree Project,we will analyzing disaster data to build a model for an API that classifies disaster messages.
In this project you will find a data set [Disaster Response Messages](https://www.figure-eight.com/dataset/combined-disaster-response-data/) from **Figure Eight** containing real messages that were sent during disaster events. These messages are sorted into 36 specific categories such as Water, Hospitals, Aid-Related, that are specifically aimed at helping emergency personnel in their aid efforts.

You will see the result on a web app where an emergency worker can input a new message and get classification results on several categories. The web app will also display visualizations of the data.

**Project Components :** There are three components we'll need to complete for this project.

**ETL Pipeline:** `process_data.py`, a data cleaning pipeline that: Loads the messages and categories datasets Merges the two datasets Cleans the data Stores it in a SQLite database

**ML Pipeline:** `train_classifier.py`, a machine learning pipeline that: Loads data from the SQLite database Splits the dataset into training and test sets Builds a text processing and machine learning pipeline Trains and tunes a model using GridSearchCV Outputs results on the test set Exports the final model as a pickle file

**Flask Web App:**`run.py` We will be taking the user message and classify them into 36 categories.

### Installation <a name="installation"></a>
For running this project,from **requirements.txt** with `pip install requirements.txt` you will install all necessary python packages for analysis and building models .

### File Descriptions <a name="files"></a>
1. data/process_data.py: The ETL pipeline used to process and clean data in preparation for model building.
2. models/train_classifier.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle.
3. app/templates/.html: HTML templates required for the web app.
4. app/run.py: To start the Python server for the web app and render visualizations.

### Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/

### Results<a name="results"></a>
The main observations of the trained classifier can be seen by running this application.
