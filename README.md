# Project 2 Disasters Pipeline
## Table of Contents
* 1.Installation
* 2.Project Motivation
* 3.File Descriptions
* 4.Licensing, Authors, and Acknowledgements
## 1.Installation
Developed under Python 3.6
Anaconda distribution distribution for Python
* Pandas, Numpy, Scikit-learn, NLP libraries from nltk, Pickle, sqlalchemy

## 2.Project Motivation
This project is a requirement of the Udacity Data Science Nanodegree. The project consists in taking two csv files merge them, clean the data, and export the cleaned data to an sqlite database. With the data we have to build a NLP pipeline for classificate the messages in 36 categories. Finally we are required to build a flask app with a couple of data vizualization and the classification model.

## 3.File Descriptions

**data** 
contains the twuo .csv files and a process.py which can be called from a terminal python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db 

**model**
contains train_classifier.py which can be called from terminal eg. python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl 

**app**
contains run.py used to deploy the flask app. It can be called from a terminal by
running the following command in the app's directory python run.py

when the app is running Go to http://0.0.0.0:3001/


## 4 Licensing, Authors, Acknowledgements

Must give credit to Udacity and its partners for the data. You can find the Licensing for the data https://www.udacity.com/course/data-scientist-nanodegree--nd025.
