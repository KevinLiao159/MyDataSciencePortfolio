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
- [Open Source NLP Libraries Demo](#open-source-nlp-libraries-demo)
  - [NLTK](#nltk-demo)
  - [Scikit-Learn](#scikit-learn-demo)
  - [Gensim](#gensim-demo)
  - [spaCy](#spacy-demo)
- [EDA & OLAP](#eda--olap)
  - [San Francisco Crime Analysis in Apache Spark](#san-francisco-crime-analysis-in-apache-spark-demo)
  - [Medium BlogPost Analysis in Pandas & Seaborn](#medium-blogpost-analysis-in-pandas--seaborn-demo)
- [Topic Modeling](#topic-modeling)
  - [NLP and Topic Modeling on Medium BlogPost with Apache Spark](#nlp-and-topic-modeling-on-medium-blogpost-with-apache-spark-demo)
  - [NLP and Topic Modeling on Medium BlogPost with Sklearn](#nlp-and-topic-modeling-on-medium-blogpost-with-sklearn-demo)
- [Recommender System](#recommender-system)
  - [Movie Recommendation Engine Development in Apache Spark](#movie-recommendation-engine-development-in-apache-spark-demo)
  - [Movie Recommendation Engine Development in Deep Learning with Keras](#movie-recommendation-engine-development-in-deep-learning-with-keras-demo)
- [Future Potential Projects](#future-potential-projects)
- [Appendix](#appendix)
  - [Source Code](#source-code)


<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Introduction
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/KevinLiao159/MyDataSciencePortfolio)
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Welcome to my awesome data science project portfolio. In my repo, you can find awesome and practical solutions to some of the real world business problems with the-state-of-art machine learning and deep learning algorithms. Most of my projects will be demoed in jupyter notebook form. Jupyter notebook is an excellent way to share my work with the world. It comes with markdown and interactive python environment and it is portable to other platforms like Databricks and Google Colaboratory as well. 

My project collection covers various trending machine learning applications such as Natural Language Processing, Large Scale Machine Learning with Spark, and Recommender System. There are more to come. Potential future projects include Text Summarization, Stock Price Forecast, Trading Strategy with Reinforcement Learning, and Computer Vision.



## Open Source NLP Libraries Demo
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


### NLTK [(DEMO)](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/nlp_intro/nltk.ipynb)
NLTK (Natural Language Toolkit) is used for such tasks as tokenization, lemmatization, stemming, parsing, POS tagging, etc. This library has tools for almost all NLP tasks. 
> Pros:
  * The earliest python NLP libraries and the most well-known full NLP library
  * Many third-party extensions
  * Supports the largest number of languages
> Cons:
  * Complicated to learn
  * Slow
  * Doesn't provide neural network models
  * No integrated word vectors

### Scikit-Learn [(DEMO)](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/nlp_intro/sklearn.ipynb)
Scikit-learn provides a large library for machine learning. The tools for text preprocessing are also presented here. 
> Pros:
  * Many functions to use bag-of-words method of creating features for text classification tasks
  * Provides a wide varity of algorithms to build ML models
  * Good documentation
> Cons:
  * Doesn't have sophisticated preprocessing things like pos-taggin, parsing, and NER
  * Doesn't use neural network models

### Gensim [(DEMO)](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/nlp_intro/gensim.ipynb)
Gensim is the package for topic and vector space modeling, document similarity.
> Pros:
  * Works with large datasets and processes data streams
  * Provides tf-idf vectorization, word2vec, document2vec, Latent Semantic Analysis, Latent Dirichlet Allocation
  * Supports deep learning
> Cons:
  * Designed primarily for unsupervised text modeling
  * Doesn't have enough tools to provide full NLP pipeline, so should be used with some other library (spaCy or NLTK)

### spaCy [(DEMO)](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/nlp_intro/spacy.ipynb)
spaCy is the main competitor of the NLTK. These two libraries can be used for the same tasks. spaCy offers a full NLP pipeline (tokenizer, tagger, parser, and NER) through spaCy's container objects such as Doc, Token, Span, and Lexeme. Compared to NLTK, spaCy is more opinionated on the architecture of a NLP pipeline.
> Pros:
  * The fastest NLP framework
  * Easy to learn and use because it has one single highly optimized tool for all tasks
  * Processes objects; object-oriented
  * Uses neural networks for training some models
  * Provides built-in word vectors
> Cons:
  * Lacks flexibility, comparing to NLTK
  * Sentence segmentation is slower than NLTK
  * Doesn't support many languages




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

### Movie Recommendation Engine Development in Deep Learning with Keras
add png for model visualization






## Future Potential Projects









## Appendix


### Source Code