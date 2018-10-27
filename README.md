<h1 align="center"> My Data Science Portfolio </h1> <br>
<p align="center">
  <a href="https://github.com/KevinLiao159/MyDataSciencePortfolio">
    <img alt="DataScience" title="DataScience" src="https://cdn-images-1.medium.com/max/1600/1*u16a0WbJeckSdi6kGD3gVA.jpeg" width="600" height="300">
  </a>
</p>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

## Table of Contents

- [Introduction](#introduction)

- [Medium Blogpost](https://github.com/KevinLiao159/MyDataSciencePortfolio/tree/master/medium_blogpost)
  - [Exploratory Data Analysis with Seaborn](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/medium_blogpost/medium_post_analysis_using_pandas.ipynb)
  - [Topic Modeling on Medium BlogPost with Sklearn](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/medium_blogpost/topic_modeling_using_sklearn.ipynb)
  - [Topic Modeling on Medium BlogPost with Apache Spark](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/medium_blogpost/topic_modeling_using_pyspark.ipynb)

- [Recommender System](https://github.com/KevinLiao159/MyDataSciencePortfolio/tree/master/recommender_system)
  - [Movie Recommendation Engine Development with Matrix Factorization]
  (#TODO)
  - [Movie Recommendation Engine Development in Apache Spark](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/recommender_system/movie_recommendation_using_ALS.ipynb)
  - [Movie Recommendation Engine Development in Deep Learning with Keras](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/recommender_system/movie_recommendation_using_NeuMF.ipynb)

- [San Francisco Crime Study](https://github.com/KevinLiao159/MyDataSciencePortfolio/tree/master/sf_crime_study)
  - [San Francisco Crime Analysis with Apache Spark](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/sf_crime_study/crimes_analysis_using_spark.ipynb)

- [Useful NLP Libraries](https://github.com/KevinLiao159/MyDataSciencePortfolio/tree/master/useful_nlp)
  - [NLTK](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/useful_nlp/nltk.ipynb)
  - [Scikit-Learn](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/useful_nlp/sklearn.ipynb)
  - [Gensim](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/useful_nlp/gensim.ipynb)
  - [spaCy](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/useful_nlp/spacy.ipynb)

- [Future Potential Projects](#future-potential-projects)
- [Appendix](#appendix)
  - [Source Code](#source-code)


<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/KevinLiao159/MyDataSciencePortfolio)
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Welcome to my awesome data science project portfolio. In my repo, you can find awesome and practical solutions to some of the real world business problems with statistical methods and the-state-of-art machine learning models. Most of my projects will be demoed in jupyter notebook. Jupyter notebook is an excellent way to share my work with the world. It comes with markdown and interactive python environment and it is portable to other platforms like Databricks and Google Colaboratory as well.

My project collection covers various trending machine learning applications such as *Natural Language Processing*, *Large Scale Machine Learning with Spark*, and *Recommender System*. There are more to come. Potential future projects include *Text Summarization*, *Stock Price Forecast*, *Trading Strategy with Reinforcement Learning*, and *Computer Vision*.


## Medium Blogpost
Medium is a popular blogpost publishing platform with enormous amount of contents and text data. What are people publishing? What are the latent topics in those blogposts? What makes a blogpost popular? And what is the trend in today's Technology? This project aims to answer the questions through visualization, analysis, natural language process, and machine learning techniques.

Specifically, we will use *Seaborn* and *Pandas* for exploratory analysis

## Recommender System



## Open Source NLP Libraries
Natural language processing (NLP) is a trending area about how to program machines to process and analyze large amounts of natural language data, and extract meaningful information from it.

I believe we are still at an early stage of NLP development. However, NLP at current stage is already able to perform many tasks. The following is a list of most commonly researched tasks in natural language processing. Note that some of these tasks have direct real-world applications.

Syntax Challenges
  * Sentence breaking
  * Word segmentation
  * Morphological segmentation
  * Stemming and Lemmatization
  * Part-of-speech tagging
  * Terminology extraction

Semantics Challenges
  * Named entity recognition (NER)
  * Relationship extraction
  * Topic segmentation and recognition
  * Sentiment analysis
  * Machine translation
  * Natural language generation
  * Question answering
  * Natural language understanding

There are many tools and libraries designed to solve NLP problems. The most commonly used libraries are Natrual Language ToolKit (NLTK), spaCy, sklearn NLP toolkit, gensim, Pattern, polyglot and many others. However, I only select four of them for demo.






## EDA & OLAP
Exploratory Data Analysis (EDA) is an approach to analyzing datasets to summarize their main characteristics, often with visual methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task. Usually in python or jupyter notebook environment, data scientists use pandas, numpy, matplotlib, seaborn or even plotly to perform EDA.

Online analytical processing (OLAP), is an approach to answering multi-dimensional analytical (MDA) queries swiftly in computing. OLAP is part of the broader category of business intelligence, which also encompasses relational databases, report writing and data mining. In the context of Big Data Analytics (Distributed Computing), data scientists often perform OLAP with SQL query on Apache License Software such as HIVE, Spark, Hadoop.

The following are two projects that I have done. One is about San Francisco Crime datasets. The other is Medium Blogpost text datasets.

### San Francisco Crime Analysis in Apache Spark [(DEMO)](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/olap/crimes_analysis_using_spark.ipynb)
* Perform analytical operations such as consolidation, drill-down, and slicing and dicing on a 15 year dataset of reported incidents from SFPD
* Perform spatial and time series analysis to further understand crime pattern and distribution in SF
* Build data processing pipeline based on Spark RDD, DataFrame, Spark SQL for various OLAP tasks
* Train and fine-tune Time Series model to forecast the number of theft incidents per month

### Medium BlogPost Analysis in Pandas & Seaborn [(DEMO)](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/olap/medium_post_analysis_using_pandas.ipynb)
* Develop statistical data visualization with seaborn to get summary statistics such as distribution of blogpost's popularity, trends in different blogpost topics, and top n popular topics and blogpost's authors
* Perform feature engineering to extract features from blogpost's contents, titles, authors, and topics
* Apply various statistical charts to understand correlation between a blogpost's popularity and its extracted features 



## Topic Modeling
[Topic modeling](https://en.wikipedia.org/wiki/Topic_model) is a type of statistical modeling for discovering the latent “topics” that occur in a collection of documents. Latent Dirichlet Allocation (LDA) is an example of topic model and is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions.

The following projects are using topic model as a text mining tool to discover the latent "topics" in Medium blogposts, as well as trends and popularity of different latent "topics". With topic modeling, we are able to identify insights about what latent 'topics' are trendy and continue to be the most popular content.

Medium blogpost datasets are scraped from [Medium](https://medium.com/) using scrapy framework. Details of scrapy implementation is in my another data science project [MediumBlog](https://github.com/KevinLiao159/MediumBlog/tree/master/src/scraper/mediumScraper)

### NLP and Topic Modeling on Medium BlogPost with Apache Spark [(DEMO)](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/topic_modeling/topic_modeling_using_pyspark.ipynb)
* Apply topic modeling to understand what drives a blog post’s popularity (as measured in claps) and the interaction between users’ preferences and blog posts’ contents
* Build a feature extraction pipeline using Spark, which consists of tokenizing raw texts, stop-words removal, stemming/lemmatization, and BOW/TF-IDF transformation
* Implement unsupervised learning models of K-means and LDA to discover latent topics embedded in blog posts and identify key words of each topics for clustering and similarity queries
* Evaluate model’s clustering results by visual displays with dimensionality reduction using PCA and T-SNE


### NLP and Topic Modeling on Medium BlogPost with Sklearn [(DEMO)](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/topic_modeling/topic_modeling_using_sklearn.ipynb)
* Perform similar tasks like above but using sklearn rather than Spark





## Recommender System
A recommender system is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item. Recommender systems are utilized in a variety of areas including movies, music, news, social tags, and products in general. Recommender systems typically produce a list of recommendations in one of two ways – through collaborative filtering or through content-based filtering.

> Collaborative filtering

This approach builds a model from a user's past behaviour (items previously purchased or selected and/or numerical ratings given to those items) as well as similar decisions made by other users. This model is then used to predict items (or ratings for items) that the user may have an interest in

> Content-based filtering

This approach utilizes a series of discrete characteristics of an item in order to recommend additional items with similar properties

> Hybrid Recommender

This one combines the previous two approaches

In my project, I will focus on building a collaborative filtering engine. In collaborative filtering, there are typically following challenges:
* cold start
* data sparsity
* popular bias (how to recommend products from the tail of product distribution)
* scalability (computation grows as number of users and items grow)
* pool relationship between like-minded yet sparse users

> Solution

Use matrix factorization technique to reduce dimensionality and sparsity, as well as capturing user information in user latent factors and item information in item latent factors

> Implementations

I choose to use two types of different ML algos to build two separate movie recommendation engines and compare their performance and results respectively. The following is the list of my ML algos to implement movie recommendation engine
* [Alternating Least Square (ALS) Matrix Factorization](https://spark.apache.org/docs/preview/ml-collaborative-filtering.html#collaborative-filtering)
* [Neural Collaborative Filtering Approach](https://arxiv.org/pdf/1708.05031.pdf)
  * Generalized Matrix Factorization (GMF)
  * Multi-Layer Perceptron (MLP)
  * Neural Matrix Factorization (NeuMF)

> Datasets

I use [MovieLens Small Datasets](https://grouplens.org/datasets/movielens/latest/). This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from MovieLens, a movie recommendation service. It contains 100004 ratings and 1296 tag applications across 9125 movies.

> Model Performance Comparison on Test Datasets

| MODEL | MEAN SQUARED ERROR | ROOT MEAN SQUARED ERROR |
| --- | --- | --- |
| ALS | 0.8475 | 0.9206 |
| - | - | - |
| GMF | 0.8532 | 0.9237 |
| MLP | 0.8270 | 0.9094 |
| NeuMF | 0.8206 | 0.9059 |


### Movie Recommendation Engine Development in Apache Spark [(DEMO)](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/recommender_system/movie_recommendation_using_ALS.ipynb)
In the context of distributed computing and large scale machine learning, Alternating Least Square (ALS) in Spark ML is definitely the one of the first go-to models for collaborative filtering in recommender system. ALS algo has been proven to be very effective for both explicit and implicit feedback datasets. 

In addition, [Alternating Least Squares with Weighted λ Regularization (ALS-WR)](https://endymecy.gitbooks.io/spark-ml-source-analysis/content/%E6%8E%A8%E8%8D%90/papers/Large-scale%20Parallel%20Collaborative%20Filtering%20the%20Netflix%20Prize.pdf) is a parallel algorithm designed for a large-scale collaborative filtering challenge, the Netflix Prize. This method is meant to resolve scalability and sparseness of the user profiles, and it's simple and scales well to very large datasets

* Advantages of collaborative filtering over content based methods
  * No need to know about item content
  * "Item cold-start" problem is avoided
  * User interest may change over time
  * Explainability

* My implementation to train the best ALS model via cross-validation and hyperparam-tuning

```python
from src.spark_recommender_system import Dataset, train_ALS
from pyspark.ml.evaluation import RegressionEvaluator

# config
SEED = 99
MAX_ITER = 10
SPLIT_RATIO = [6, 2, 2]
DATAPATH = './data/movie/ratings.csv'

# construct movie ratings dataset object
rating_data = Dataset(spark, DATAPATH)
# get rating data as Spark RDD
rating_rdd = rating_data.RDD
# get train, validation, and test data
train_data, validation_data, test_data = rating_data.split_data(rating_rdd, SPLIT_RATIO, SEED)
# create a hyperparam tuning grid
regParams = [0.05, 0.1, 0.2, 0.4, 0.8]
ranks = [6, 8, 10, 12, 14]
# train models and select the best model in hyperparam tuning
best_model = train_ALS(train_data, validation_data, MAX_ITER, regParams, ranks)
# test model
predictions = best_model.transform(test_data)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
```

### Movie Recommendation Engine Development in Neural Networks with Keras [(DEMO)](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/recommender_system/movie_recommendation_using_NeuMF.ipynb)
[Neural Collaborative Filtering (NCF)]((https://arxiv.org/pdf/1708.05031.pdf)) is a paper published by National University of Singapore, Columbia University, Shandong University, and Texas A&M University in 2017. It utilizes the flexibility, complexity, and non-linearity of Neural Network to build a recommender system. It proves that Matrix Factorization, a traditional recommender system, is a special case of Neural Collaborative Filtering. In addition, it shows that NCF outperforms the state-of-the-art models in two public datasets

Before we get into the Keras implementation of Neural Collaborative Filtering (NCF), let's quickly review Matrix Factorization and how it is implemented in the context of Neural Networks.

Here is the illustration of the math:
![Matrix factorization](https://cdn-images-1.medium.com/max/2000/1*WrOoSr49lQs43auSsLlLdg.png)

Essentially, each user and item is projected onto a latent space, represented by a latent vector. The more similar the two latent vectors are, the more related the corresponding users’ preference. Since we factorize the user-item matrix into the same latent space, we can measure the similarity of any two latent vectors with cosine-similarity or dot product.

In Neural Network, we will implement an embedding layer. We usually map a user one-hot encoded vector to a user embedded vector, and map a item one-hot encoded vector to an item vector. Then we will do a element-wise multiplication between user latent vector and item latent vector. Now we have a element-wise user-item latent vector. 

In traditional matrix factorization, we would just sum up the vector, which is also the dot product of user latent vector and item latent vector. Then we minimize the loss between the dot product of these two and the true ratings in user-item association matrix

However, in the world of neural network, we can generalize matrix factorization by feeding the element-wise user-item latent vector into FC layer. The Neural FC layer can be any kind neuron connections. With the complicated connection and non-linearity in the Neural CF layers, this model is capable of properly estimating the complex interactions between user and item in the latent space. Then the objective function is to minimize the loss between the predictions and the ratings. This is exactly how Generalized Matrix Factorization (GMF) is implemented. Below is the graph of network architecture:
![Generalized Matrix Factorization (GMF)](https://cdn-images-1.medium.com/max/2000/1*EA03sZsfJ4wu8yMoU6xwPQ.png)

To further generalize the process of matrix factorization in neural network, we need to increase the complexity in hypothesis space of the network and remove calucation rules from the neural topology. This means we will remove the element-wise multiplication layer and add more Neural CF layers, for example, multiple layer perceptron (MLP), can be placed after the concat layer of user and item embedded layers. And this is the Multi-Layer Perceptron (MLP) model. Below is the graph of network architecture:
![Multi-Layer Perceptron (MLP)](https://cdn-images-1.medium.com/max/1600/1*sTBtqrsQzTKlZ8hSU7I6FQ.png)

Now that we understand how generalized matrix factorization works in the world of neural network, the next question is how we can improve the model. One simple trick that is often used in Machine Learning competitions is "stacking". In neural networks, "stacking" means we concat the outputs of GMF and MLP networks and connect it with the sigmoid activation output layer. And this is Neural Matrix Factorization (NeuMF). Below is the graph of network architecture:
![Neural Matrix Factorization (NeuMF)](https://cdn-images-1.medium.com/max/1600/1*CoETyuU36fshduKAfFhCrg.png)

* My implementation to build Neural Matrix Factorization (NeuMF) and train the model


```python
import pandas as pd
from src.neural_recommender_system import (get_GMF_model,
                                           get_MLP_model,
                                           get_NeuMF_model,
                                           train_model,
                                           load_trained_model)
# data config
DATAPATH = './data/movie/ratings.csv'
MODELPATH = './data/movie/tmp/model.hdf5'
SEED = 99
TEST_SIZE = 0.2

# model config
EMBEDDED_DIM = 10
L2_REG = 0
MLP_HIDDEN_LAYERS = [64, 32, 16, 8]

# trainer config
OPTIMIZER = 'adam'
BATCH_SIZE = 64
EPOCHS = 30
VAL_SPLIT = 0.25

# load ratings
df_ratings = pd.read_csv(
    DATAPATH,
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

# get total number of users and items
num_users = len(df_ratings.userId.unique())
num_items = len(df_ratings.movieId.unique())

# train/test split
df_train, df_test = train_test_split(df_ratings, TEST_SIZE, SEED)

# build Generalized Matrix Factorization (GMF)
GMF_model = get_GMF_model(num_users, num_items, EMBEDDED_DIM, L2_REG, L2_REG)

# build Multi-Layer Perceptron (MLP)
MLP_model = get_MLP_model(num_users, num_items, 
                          MLP_HIDDEN_LAYERS, [L2_REG for i in range(4)])

# build Neural Matrix Factorization (NeuMF)
NeuMF_model = get_NeuMF_model(num_users, num_items, EMBEDDED_DIM,
                              (L2_REG, L2_REG), MLP_HIDDEN_LAYERS, 
                              [L2_REG for i in range(4)])

# let's just train Neural Matrix Factorization (NeuMF)
train_model(NeuMF_model, OPTIMIZER, BATCH_SIZE, EPOCHS, VAL_SPLIT, 
            inputs=[df_train.userId.values, df_train.movieId.values], 
            outputs=df_train.rating.values,
            filepath=MODELPATH)

# load the best trained model
# rebuild
NeuMF_model = get_NeuMF_model(num_users, num_items, EMBEDDED_DIM,
                              (L2_REG, L2_REG), MLP_HIDDEN_LAYERS, 
                              [L2_REG for i in range(4)])
# load weights
NeuMF_model = load_trained_model(NeuMF_model, MODELPATH)

# define metric - rmse
rmse = lambda true, pred: np.sqrt(
  np.mean(
    np.square(
      np.squeeze(predictions) - np.squeeze(df_test.rating.values)
    )
  )
)

# test model
predictions = NeuMF_model.predict([df_test.userId.values, df_test.movieId.values])
error = rmse(df_test.rating.values, predictions)
print('The out-of-sample RMSE of rating predictions is', round(error, 4))
```


## Future Potential Projects









## Appendix


### Source Code