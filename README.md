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

the project uses data about messages collected by figure eigth about messages recived during natural disasters. The project will build an NLP model to help classify the messages in order to improve the response capacity of emergency services.

## 3.File Descriptions

**data** 
contains the twuo .csv files and a process.py which can be called from a terminal "python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db". The data folder contains the following files:
* **disaster_messages** Contains the id, message that was sent and genre i.e the method (direct, tweet..) the message was sent
* **disaster_categories** Contains the id and the categories (related, offer, medical assistance..) the message belonged to.
* **process_data.py** ETL that reads, merge and clean the data.
* **ETL Pipeline Preparation.ipynb** jupiter notebook with the process.
* **DisasterResponse.db** sqlite database to store de cleaned data.

**models**
contains train_classifier.py which can be called from terminal  "python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl" . The models folder contains the following files:
* **process_data.py** pipeline that trains the NLP model and _creates the **classifier.pkl** in the model folder (because of its size it can't be uploaded)._
* **ETL Pipeline Preparation.ipynb** jupiter notebook with the process.

**app**
contains **run.py** used to deploy the flask app. It can be called from a terminal by running the following command in the **app's directory** "python run.py" (before running make sure to be in the app directory use **cd app** in the terminal.

when the app is running Go to http://0.0.0.0:3001/


## 4 Licensing, Authors, Acknowledgements

Must give credit to Udacity and its partners for the data. You can find the Licensing for the data https://www.udacity.com/course/data-scientist-nanodegree--nd025.
