# Disaster Response Pipeline Project

## Short Summary of the project and its objectives:

- Project contains a classifier used for categorizing messages with regards to disaster situations.
- The project has a significant importance when it comes to its utility, as the impact on helping classifying the distater messages will ensure a faster response from the authorities and maybe to more lives saved, in the end within the community.
- In case of a disaster, the response time of authorities is critical. This project helps exactly in this regard.


##  Instructions on how to prepare the data and launch the app:

#### For preparing the data and training the model: (terminal is needed)

1. Run the following commands in the project's root directory to set up the database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

#### For launching the app:

2. Go to `app` directory: `cd app`
3. Run your web app: `python run.py`
4. Click the `PREVIEW` button to open the homepage



## Summary of the package:

#### Code used in preprocesisng and preparing the data:
- |---> ETL Pipeline Preparation.ipynb (pure Python Notebook)
- |---> process_data.py (app file with functions built based on the Python Notebook)

#### FILES containing the raw data:
- |---> disaster_categories.csv
- |---> disaster_messages.csv

#### Output of the data preprocesisng:
- |---> DisasterResponse.db


#### Code used in training the model:
- |---> ML Pipeline Preparation.ipynb (pure Python Notebook)
- |---> train_classifier.py (app file with functions built based on the Python Notebook)

#### Output of the model train:
- |---> classifier.pkl

#### Files used to launch app:
-|---> run.py

#### Requirements file:
- |---> requirements.txt


## Explanation of the files in the package

##

### 1. ETL Pipeline Preparation.ipynb
Jupyter notebook containing the data processing code.
### 2. process_data.py
File containing the functions for cleaning up the data and creating an sqlite DB along with a table. Built based on the Python code in the ETL Pipeline Preparation.ipynb.
### 3. DisasterResponse.db
SQLITE Database containing the message table cleaned and pre-processed for the messages and their categories.

## -------------------------------------------------------------------------------------------------------

### 4. disaster_categories.csv
File containing the types of categories each message falls into.
### 5. disaster_messages.csv
File containing the actual message data, along with the ids of categories.

## -------------------------------------------------------------------------------------------------------

### 6. ML Pipeline Preparation.ipynb
Jupyter notebook containing the code for the creation of the ML algorithm: training, testing, evaluation.
### 7. train_classifier.py
File containing the functions for training and evaluation the Decision Tree Classifier. Built based on the Python code in the ML Pipeline Preparation.ipynb.
### 8. classifier.pkl
PKL File containing the trained and evaluated model.

## -------------------------------------------------------------------------------------------------------

### 9. run.py
File that runs the application, along with code for the graphs on the page.

## -------------------------------------------------------------------------------------------------------

### 10. requirements.txt
File containing all package requirements necessary for running this app.
