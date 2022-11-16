# SageMaker Deployment Project

The notebook and Python files provided here, once completed, result in a simple web app which interacts with a deployed recurrent neural network performing sentiment analysis on movie reviews. This project assumes some familiarity with SageMaker, the mini-project, Sentiment Analysis using XGBoost, should provide enough background.

Please see the [README](https://github.com/udacity/sagemaker-deployment/tree/master/README.md) in the root directory for instructions on setting up a SageMaker notebook and downloading the project files (as well as the other notebooks).

## Short Summary of the project and its objectives:

- Project contains a classifier used for categorizing messages with regards to movie reviews.
- Project was built using Amazon Sagemaker deployment environment.
- The project is a fun and interactive way of trying to predict is a movie review is positive or negative.


##  Instructions on how to prepare the data and launch the app:

#### For preparing the data and training the model: (terminal is needed)

1. Go to the website folder
2. Click on index.html
3. Input a review
4. The result should show POSITIVE or NEGATIVE



## Summary of the main files within the package:

#### Code used in preprocesisng, preparing the data, training and deploying model, and testing new inputs:
- |---> SageMaker Project.ipynb

#### FILE containing the definition of model used for doing inference for the user reviews
- |---> model.py

#### FILE containing the methods used for training the model:
- |---> train.py

#### FILE containing the methods used by the model for doing inference on the user reviews:
- |---> predict.py


#### FILE containing the method for data pre-processing into a form compatible with our model:
- |---> utils.py

#### HTL Page where users can input the review, and get inference back (prediction):
- |---> index.html

#### Requirements file:
- |---> requirements.txt


## Summary of the of the AWS SageMaker objects:

#### Lambda Function:
- |---> sentiment_lambda_function_lupascu

#### Lambda Function Role:
- |---> LambdaSageMakerRoleLupascu

#### API Gateway:
- |---> SentimentAnalysisLupascu URL https://8ran5jdojl.execute-api.us-east-1.amazonaws.com/PROD

#### DEPLOYMENT ENDPOINT 
- |---> could not deploy due to CUDA library error.


