# Recommendation Engines - IBM

## Short Summary of the project and its objectives:

- Project uses Singular Value Decomposition to try and make user recommendations for articles that would be within their interest.
- Recommendations are created based on user prefernces, along with taking into consideration other techniques as well, especially for new users.
- Final scope of the project is to create the best recommendations for users (good recommendations that would be followed through for both old and new users).

##  Project Sections:

#### 

I. Exploratory Data Analysis
II. Rank Based Recommendations
III. User-User Based Collaborative Filtering
IV. Content Based Recommendations (not filled in, unfortunately)
V. Matrix Factorization


## Summary of the files:

#### Files containing raw data:
- |---> articles_community.csv : contains short descriptions of the articles included in the data set, along with the article_id;
- |---> user-item-interactions.csv : main file used for this analysis and recommendation engine: contains data about the interactions between the users and certain articles.

#### Python notebook with entire code:
- |---> Recommendations_with_IBM.ipynb

#### Other files for validation of exercises:
- |---> project_tests.py: file with various functions for exercise validations
- |---> top_5.p
- |---> top_10.p
- |---> top_20.p
