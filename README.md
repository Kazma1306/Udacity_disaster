# Description 
In this project, the task is to analyze disaster data and form an api. 
In this training, all tweets are natural disaster tweets.

# There are three steps:

* Create an ETL which cleans the Data
* Create a ML pipeline which performs feature extraction and trains a model
* Take model and embed it into a webapp


# Packages Used
* sys
* pandas
* sqlalchemy
* re
* nltk
* json
* flask

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
