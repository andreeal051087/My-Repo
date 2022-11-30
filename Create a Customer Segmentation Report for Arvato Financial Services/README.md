
# Project Definition

Blog source can be found [here](https://andreealupascu0510.wixsite.com/customersegmentation/post/identifying-customer-segments-arvato-use-case)

Through this project I am trying to determine which what type of customer typology we are able to determine form the data provided by the mail-order company, in order to establish the most likey and unlikely people to purchase their products.

I have personally chosen this project due to the fact that I find it very interesting and has a real-life application in any industry one chooses (or ends up) working in.

Being able to do a proper customer segmentation and targeting is the foundation for any successful business (other than having a good product to begin with). From personal experience, companies are investing a lot of resources in trying to come up with the best and most close-to-reality customer archetypes, in order to be able to better concentrate their resources when it comes to branded/non-branded advertising and building a healthy pipeline when it comes to the customer universe.

I will use 2 algorithms for :

1. Unsupervised modelling: determinig clusters by using PCA and KMeans
2. Supervised learning: by using XGBoost and GradientBoosting.

The unsupervised modelling uses 2 very common techniques: PCA: principal component analysis- in order to perform dimensionality reduction, so that we may group all te features that seem to be somewhat correlated together.

The supervised methods is using 2 powerful algortihms named XGBoost, and also GradientBoosting - the choice for these two comes mostly form the fact that the training data we will work with is very sparse. The response data coming from customers is not so well defined, therefore we need to try models that perform well when we don't have enough data.

Gradient boosting is a supervised learning algorithm, which attempts to accurately predict a target variable by combining the estimates of a set of simpler, weaker models.

We need to use models that would learn form previous weaker iterations..

I will describe however the metics used in the next parts of the project.


# Project Structure:

## Part 1: Project Definition
As mentioned.

## Part 2: Analysis

- Structure
- Data types
- Data values balance (meand, std)

### Step 2.1: Preliminary comparison of the demographics data vs customers one in terms of rough individual groups.
### Step 2.2: Assess Structure and  Data Types


## Part 3A: Methodology: Data Preprocessing
### Step 3.1: Assess Missing Data Overall - Data Preprocessing
----Step 3.1.1: Assess Missing Data in Each Column and dropping problematic features.
----Step 3.1.2: Assess Missing Data in Each Row and dropping problematic rows.
### Step 3.2: Select and Re-Encode Features (categorical and mixed)
----Step 3.2.1 Re-Encode Categorical Features (for general population data)
----Step 3.2.2 Engineer Mixed-Type Features (for general population data)
### Step 3.3: Create a Cleaning Function


## Part 3B: Customer Segmentation Report: Implementation
### Step 3.1 : Feature Transformation
----Step 3.1.1: Apply Feature Scaling
----Step 3.1.2: Perform Dimensionality Reduction
----Step 3.1.3: Function for Principal Components check
### Step 3.2: Clustering
----Step 3.2.1: Apply Clustering to General Population Data
----Step 3.2.2: Apply Clustering to Customer Data
### Step 3.3: Compare Customer Data to General Population Data

## Part 4: Supervised Learning Model: Model Evaluation and Validation
### Step 4.1: XGBoost
### Step 4.2: GradientBoosting
### Step 4.3: Predicting on TEST data- submission for Kaggle competition

## Part 5: Conclusions



# Files in this folder:
----> Arvato Project Workbook.ipynb : notebook containing project;
----> terms_completed.md: terms and conditions that I have accepted, stating I have deleted any copy on the local machine or on personal or work directories of the data;
--- > requirements.txt: a file containing all libraries that were used for the project;
---- lupascu_gradientboost_submission.csv : Kaggle submission which I was not able to submit.


## Part 6: References and Acknowledgements:

Luckily, the, internet is a vast place filled with resources and other people's shared knowledge- a data -learning wonderful place.
Therefore, I have managed to compile a pretty long list of References that helped me in creating the project.

### List of References is below:

Luckily, the, internet is a vast place filled with resources and other people's shared knowledge- a data -learning wonderful place.
Therefore, I have managed to compile a pretty long list of References that helped me in creating the project.

Reference list: (online tools and forums)

For Data pre-processing and analysis:
https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
https://stackoverflow.com/questions/29803093/check-which-columns-in-dataframe-are-categorical 
https://www.w3schools.com/python/python_try_except.asp 
https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python 
https://numpy.org/doc/stable/reference/generated/numpy.sum.html
https://pandas.pydata.org/docs/reference/api/pandas.unique.html


For unsupervised learning methods and process:
https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html 
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html 
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html 
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html 
https://www.geeksforgeeks.org/python-ways-to-sort-a-zipped-list-by-values/ 
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans 
https://seaborn.pydata.org/generated/seaborn.barplot.html 
https://stackabuse.com/k-means-clustering-with-scikit-learn/ 

For supervised learning methods and process:
https://xgboost.readthedocs.io/en/stable/python/python_intro.html 
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html 
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html 
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html 
https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ 
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html 
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html 
https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters 
https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/ 

### List of Acknowledgements is below:

And of course, I have lots of acknowledgments to give:

Below courses that have helped me a lot. All of the below resources can be found on [UDACITY website](https://www.udacity.com/):
- UDACITY Unsupervised Learning course within Data Science Foundations: GMM Clustering and Cluster Validation Lab 
- UDACITY Unsupervised Learning course within Data Science Foundations: Hierarchical Clustering Lab
- UDACITY Unsupervised Learning course within Data Science Foundations: Interpret PCA Results
- UDACITY Unsupervised Learning course within Data Science Foundations: Random Projection Lab
- UDACITY course Machine Learning in Production within Data Science Program: IMDB Sentiment Analysis - XGBoost (Hyperparameter Tuning)
- UDACITY course Experimental Design and Recommendations / Matrix Factorization for Recommendations: Singular Value Decomposition

For creating the Blog- anyone can do so by accessing WiX:
https://www.wix.com 

As well, I would like to also thank:

- Everton Bin  (can be reached at https://www.linkedin.com/in/evertonbin/) who has given me the inspiration on how to create my first ever Blog post.
- Anand Kumpatla (Sr Data Scientist @ Doubleslash Software Solutions Pvt Ltd) for the very good explanation of the correlation matrix explanation and example on https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python 
- Aniruddha Bhandari (Author: Since February 2020 on the https://www.analyticsvidhya.com/blog/author/aniruddha/ site) for AUC-ROC Curve visual explanation and refresher on the calculation

