#  About this project
The purpose of this project is to put into practice the lessons learned in the Machine Learning with Python course from IBM on the [Coursera](https://www.coursera.org/learn/machine-learning-with-python?specialization=ibm-data-science) platform.
In each notebook will be analyzed different types of machine learning through different problems and create models, evaluate their accuracy with different techniques and compare them. The databases chosen for this project were: [Life Insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance) for regression, [Heart Failure](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) for classification and [Automobile Company](https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation) for clustering.

# How this project is organized
The notebooks in this repository are based on a structured sequence: starting with a brief introduction of the problem, load libraries and data, exploratory data analysis,  data preprocessing, models brief explanation , models comparison and evaluation,  references used.

About the datasets, it was chosen simple ones - with no missing values and few data wrangling - just for sake of simplicity.

## What is machine learning and Why it is important?

Machine Learning (ML) is a field of artificial intelligence (AI) that allows computers to 'learn'  through pattern recognition how to (theoretically) predict outcomes without being explicitly programmed to do so.
Some common application fields and industries of machine learning are:

- Finance: predict a sales person's total yearly sales, price estimation, predict employment income;
- Retail/Marketing: identifying buying patterns, recommending new products, predict whether a customer will switch brand;
- Banking: fraud detection in credit card, characterizing customer behavior;
- Insurance:fraud detection in claims analysis, insurance risk of customers;
- Medicine: characterizing patient behavior, predict individual satisfaction ;

## Types of machine learning
Each one of the items below is a way that an algorithm can 'learn' to make more accurate predictions. There are three main basic approaches:supervised learning, unsupervised learning and reinforcement learning. The choice of model will depend on the problem to be solved or what you want to predict.

### Supervised learning
In this type of machine learning, the algorithm learns a function based on data variables to make prediction of a defined label based on the input data. Also, the input and the output of the algorithm are specified to the model. The learning of the process can be classification or regression.

- `Classification`: process of categorizing a given set of data into classes. It can be a binary classification (yes/no or 1/0) or a multiclass classification (more than 2 categories).
Some popular algorithms are K-Nearest Neighbor(KNN), Logistic Regression, Support Vector Machines and Decision Trees/Random Forest.
- `Regression`: process of forecasting and outcome, predicts continuous values. It can be a simple regression (two variables) or a multiple regression (more than 2 variables). Also, these models can be linear and non-linear.
Some popular algorithms are Linear Regression, Polynomial Regression, Principal Components Regression (PCR) and also Decision Trees/Random Forest.

### Unsupervised learning
In this type of machine learning consists of training a machine from data that is unlabeled and/or classified. The algorithm looks for patterns in the data and group them according to similar characteristics, and the most popular type of doing so is `Clustering`. Some clustering algorithms are:
- Partitioned-based clustering is a group of clustering algorithms that produces sphere-like clusters, such as: K-Means, K-Medians or Fuzzy c-Means.
- Hierarchical clustering algorithms produce tree of clusters, such as agglomerative and divisive algorithms.
- Density-based clustering algorithms produce arbitrary shaped clusters, such DBSCAN.

It is worth noting that clustering is not the only type of unsupervised learning, dimensionality reduction is another one, it reduces the number of variables in a dataset without compromising the model's performance.

### Reinforcement learning
Reinforcement learning is the training of machine learning models to make a sequence of decisions. The agent learns to accomplish a goal in an uncertain and potentially complex environment. In reinforcement learning, the AI system is faced with a situation. The computer uses trial and error to find a solution to the problem. To make the machine do what the programmer wants, the artificial intelligence receives rewards or penalties for the actions it performs. Its objective is to maximize the total reward.

## References

* [Machine Learning with Python - Introduction to Machine Learning | Coursera](https://www.coursera.org/learn/machine-learning-with-python)
* [What Is Machine Learning and Why Is It Important?](https://www.techtarget.com/searchenterpriseai/definition/machine-learning-ML)
