
# Disaster Response Pipeline Project
### Table of Contents
1. [Project Motivation](#motivation)
2. [Dataset](#dataset)
3. [File Structure](#file)
4. [Libraries and Dependencies](#libraries)
5. [Instruction for running application](#instructions)
6. [Results](#results)

## Project Motivation<a name="motivation"></a>

This project is a part of the Data Science NanoDegree program. The goal of this project is to implement a disaster response app to categorize messages received during disasters such as earthquakes, flooding, and so on. Each message is classified into certain categories, thus assisting the disaster response team to aliquote and provide help to the right place at the right time. Following data science techniques were used in this project:

1. ETL pipeline - Implement an ETL pipeline that reads two CSV files (*message.csv* and *categories.csv*), cleans the data, and merges two files into a single dataset. After merging,  the dataset is stored in an SQLite database.

2. ML pipeline - Split the data into a training set and a test set. Then, create a machine learning pipeline that uses NLTK, as well as Scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (*multi-output classification*). Finally, the model is exported as a pickle file that will be used in the flask app. 

3. Flask web app - an application to display the result of the model.


## Dataset<a name="dataset"></a>

The dataset used in this project is taken from [Figure Eight](https://appen.com/). It contains over 26000 real messages that were obtained during a real crisis. It consists of two files:
1. message.csv - messaged received during the crisis
2. categories.cvs - categories of each message
       
File Structure<a name="file"></a>
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app`

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model
```

## Libraries and Dependencies<a name="libraries"></a>

Following libraries were used in this project.

 - python
 - sikit-learn
 - pandas
 - numpy
 - nltk
 - flask
 - sqlalchemy
 - plotly

## Instruction for running application<a name="instructions"></a>

In order to run this application, please execute the following steps
   1. To run the ETL pipeline to create train ready dataset.
   ```
   python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
   ```
   2. To train the model and save it 
   ```
   python train_classifier.py ../data/DisasterResponse.db classifier.pkl
   ```   
   3. To run the web locally,
   ```
   python run.py
   ```
   
 ## Results
 The primary goal of this project was to implement data engineering and data science skills to build a full ETL and ML pipeline. Thus, the minimum effort was put in to optimize the performance of the classifier. Based on my previous project experience, I selected the AdaBoostClassier model. I used Sklearn GridSearchCV to fine-tune parameters for this classifier. However, due to limitations in computational resources, very few parameters were fine-tuned in this project. To evaluate the model, accuracy, precision, and recall for each category are analyzed and displayed.