# Disaster Response Pipeline Project
Data Engineering Section -- Project: Disaster Response Pipeline

## Table of Contents
1. [Project Motivation](#ProjectMotivation)
1. [Installation](#installation)
2. [Instructions](#instructions)
3. [File Descriptions](#files)
4. [Results](#results)

## Project Motivation <a name="ProjectMotivation"></a>
Figure Eight Data Set:  [Disaster Response Messages](https://www.figure-eight.com/dataset/combined-disaster-response-data/) provides thousands of messages that have been sorted into 36 categories. These messages are sorted into specific categories such as Water, Hospitals, Aid-Related, that are specifically aimed at helping emergency personnel in their aid efforts.

The main goal of this project is to build an app that can help emergency workers analyze incoming messages and sort them into specific categories to speed up aid and contribute to more efficient distribution of people and other resources.

### Installation <a name="installation"></a>
For running this project, the most important library is Python version of Anaconda Distribution. It installs all necessary packages for analysis and building models.


### Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/


### File Descriptions <a name="files"></a>
1. data/process_data.py: The ETL pipeline used to process and clean data in preparation for model building.
2. models/train_classifier.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle.
3. app/templates/*.html: HTML templates required for the web app.
4. app/run.py: To start the Python server for the web app and render visualizations.

### Results<a name="results"></a>
The main observations of the trained classifier can be seen by running this application.
