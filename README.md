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
  - [NLP and Topic Modeling on Medium BlogPost with Apache Spark](#nlp-and-topic-modeling-on-medium-blogpost-with-apache-spark)
  - [NLP and Topic Modeling on Medium BlogPost with Sklearn](#nlp-and-topic-modeling-on-medium-blogpost-with-sklearn)
- [Recommender System](#recommender-system)
  - [Movie Recommendation Engine Development in Apache Spark](#movie-recommendation-engine-development-in-apache-spark)
  - [Movie Recommendation Engine Development in Deep Learning with Keras](#movie-recommendation-engine-development-in-deep-learning-with-keras)
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
* Build various chart




## Topic Modeling




### NLP and Topic Modeling on Medium BlogPost with Apache Spark




### NLP and Topic Modeling on Medium BlogPost with Sklearn








## Recommender System




### Movie Recommendation Engine Development in Apache Spark



### Movie Recommendation Engine Development in Deep Learning with Keras







## Future Potential Projects









## Appendix


### Source Code