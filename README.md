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

- [Customer Churn Study](https://github.com/KevinLiao159/MyDataSciencePortfolio/tree/master/churn_study)
  - [Customer Churn Modeling](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/churn_study/customer_churn_modeling.ipynb)

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

- [Synopsis Clustering](https://github.com/KevinLiao159/MyDataSciencePortfolio/tree/master/synopsis_clustering)
  - [Synopsis Clustering](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/synopsis_clustering/synopsis_clustering.ipynb)

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

## Customer Churn Study
<p align="center">
  <a href="https://github.com/KevinLiao159/MyDataSciencePortfolio/tree/master/churn_study">
    <img alt="Customer Churn Study" title="Customer Churn Study" src="https://glideconsultingllc.com/wp-content/uploads/2017/02/customer-journey.png">
  </a>
</p>

Churn rate is one of the important business metrics. A company can compare its churn and growth rates to determine if there was overall growth or loss. When the churn rate is higher than the growth rate, the company has experienced a loss in its customer base.

Why customers churn and stop using a company's services? What is the estimate amount of churn for next quarter? Being able to answer above two questions can provide meaningful insights about what direction the company is currently heading towards and how the company can improve its products and services so that constomers would stay. 

## Medium Blogpost
<p align="center">
  <a href="https://github.com/KevinLiao159/MyDataSciencePortfolio/tree/master/medium_blogpost">
    <img alt="Medium Blogpost" title="Medium Blogpost" src="http://yosinski.com/mlss12/media/slides/MLSS-2012-Blei-Probabilistic-Topic-Models_020.png">
  </a>
</p>

Medium is a popular blogpost publishing platform with enormous amount of contents and text data. What are people publishing? What are the latent topics in those blogposts? What makes a blogpost popular? And what is the trend in today's Technology? This project aims to answer the questions through visualization, analysis, natural language process, and machine learning techniques.

Specifically, I will use **Seaborn** and **Pandas** for exploratory analysis. For machine learning modeling, I choose **K-means**, **tSVD**, and **LatentDirichletAllocation** for topic modeling. I will perform this study with two different ML frameworks: **Sklearn** and **Spark**.

**Sklearn** is a great python machine learning library for data scientist.

However, in the age of Big Data, most data analysis are predicated on distributed computing. **Spark** is distributed cluster-computing framework and provides an interface for programming entire clusters with implicit data parallelism and fault tolerance.


## Recommender System
<p align="center">
  <a href="https://github.com/KevinLiao159/MyDataSciencePortfolio/tree/master/recommender_system">
    <img alt="Recommender System" title="Recommender System" src="http://datameetsmedia.com/wp-content/uploads/2018/05/2ebah6c-1.png">
  </a>
</p>

Most products we use today are powered by recommendation engines. Youtube, Netflix, Amazon, Pinterest, and long list of other data products all rely on recommendation engines to filter millions of contents and make personalized recommendations to their users.

It'd be so cool to build a recommender system myself. In generaly, recommender systems can be loosely broken down into three categories: **content based systems**, **collaborative filtering systems**, and **hybrid systems** (which use a combination of the other two).

My project focuses on collaborative filtering systems. Collaborative filtering based systems use the actions of users to recommend other items. In general, they can either be user based or item based. Item-based approach is usually prefered than user-based approach. User-based approach is often harder to scale because of the dynamic nature of users, whereas items usually don't change much, so item-based approach often can be computed offline.

However, both item-based and user-based collaborative filtering still face following challenges:
* cold start
* data sparsity
* popular bias (how to recommend products from the tail of product distribution)
* scalability

To overcome above challenges, I will use **Matrix Factorization** to learn latent features and interaction between users and items 


## San Francisco Crime Study
<p align="center">
  <a href="https://github.com/KevinLiao159/MyDataSciencePortfolio/tree/master/sf_crime_study">
    <img alt="San Francisco Crime Study" title="San Francisco Crime Study" src="https://support.trulia.com/hc/article_attachments/360001824668/blobid0.png">
  </a>
</p>

San Francisco has been arising as one the most expensive city to reside. More and more startups and companies move in the city and attracts more and more talents into the city. However, the crime incidents seem to rise up as the average income of its residents too. Car break-ins hit 'epidemic' levels in San Francisco. 

In this study, I will use **Spark** to analyze a 15-year reported incidents dataset from SFPD, and use machine learning methods to understand crime pattern and distribution in SF. Lastly, I will build a time-series forecast model to forecast crime rate


## Synopsis Clustering
<p align="center">
  <a href="https://github.com/KevinLiao159/MyDataSciencePortfolio/tree/master/synopsis_clustering">
    <img alt="Synopsis Clustering" title="Synopsis Clustering" src="https://juliasilge.com/blog/2018/2018-01-25-sherlock-holmes-stm_files/figure-html/unnamed-chunk-6-1.png">
  </a>
</p>

Today, we can collect a lot more unstructured data then ever before. Unlike structured data, unstructured data is not structured via pre-defined data models or schema, but it does have internal structure. One example of unstructured data is text data, such as plot summary, synopsis of movies.  

In this project, I will use classical **NLP** techniques: **word tokenization**, **word stemming**, **stopword removal**, **TF-IDF** and more to clean raw text data and extract features from raw text. Then I will use unsupervised learning models such as **K-means** and **LatentDirichletAllocation** to cluster unlabeled documents into different groups, visualize the results and identify their latent topics/structures. 

With clustering techniques applied to unstructured data, we can start to discover the internal structure inside the data and identify similarity between documents. With the similarity score between documents, we start to have the ability to query and analyze documents from any document store.


## Useful Open Source NLP Libraries
<p align="center">
  <a href="https://github.com/KevinLiao159/MyDataSciencePortfolio/tree/master/useful_nlp">
    <img alt="Useful Open Source NLP Libraries" title="Useful Open Source NLP Libraries" src="https://i.pinimg.com/originals/ef/c1/0f/efc10ffb5cee7a5c8d76b3a229ae5a86.png">
  </a>
</p>

Natural language processing (NLP) is a trending area about how to program machines to process and analyze large amounts of natural language data, and extract meaningful information from it.

There are many tools and libraries designed to solve NLP problems. The most commonly used libraries are **Natrual Language ToolKit (NLTK)**, **spaCy**, **sklearn NLP toolkit**, **gensim**, **Pattern**, **polyglot** and many others. My notebook will introduce the basic usage, pros and cons of each NLP libraries. 


## Future Potential Projects

## Appendix
### Source Code
