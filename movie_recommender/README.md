<h1 align="center"> Movie Recommender Systems </h1> <br>
<p align="center">
  <a href="https://s3.amazonaws.com/re-work-production/post_images/524/netflixf/original.png?1519061395">
    <img alt="Recommender Systems" title="Recommender Systems" src="https://s3.amazonaws.com/re-work-production/post_images/524/netflixf/original.png?1519061395">
  </a>
</p>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

## Contents
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)

- [Recommender System Overview](#recommender-system-verview)
   - [Collaborative Filtering](#collaborative-filtering)
   - [Content-based Filtering](#content-based-filtering)
   - [Hybrid Recommender](#hybrid-recommender)
   - [Common Challenges](#common-challenges)
   - [Solution](#solution)

- [Movie Recommender System Development](#movie-recommender-system-development)

  - [Movie Recommendation Engine Development with KNN](#movie-recommendation-engine-development-with-knn)
  - [Movie Recommendation Engine Development with ALS in Apache Spark](#movie-recommendation-engine-development-with-als-in-apache-spark)
  - [Movie Recommendation Engine Development with Neural Networks in Keras](#movie-recommendation-engine-development-with-neural-networks-in-keras)

- [Source Code](https://github.com/KevinLiao159/MyDataSciencePortfolio/tree/master/movie_recommender/src)
  - [Movie Recommender with KNN Item Based Collaborative Filtering](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/movie_recommender/src/knn_recommender.py)
  - [Movie Recommender with Alternative Least Square Matrix Factorization](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/movie_recommender/src/als_recommender.py)
  - [Movie Recommender with Neural Collaborative Filtering](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/movie_recommender/src/neural_recommender.py)

- [References](#references)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Recommender System Overview
A recommender system is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item. Recommender systems are utilized in a variety of areas including movies, music, news, social tags, and products in general. Recommender systems typically produce a list of recommendations in one of two ways – through collaborative filtering or through content-based filtering.

#### Collaborative Filtering
This approach builds a model from a user's past behaviour (items previously purchased or selected and/or numerical ratings given to those items) as well as similar decisions made by other users. This model is then used to predict items (or ratings for items) that the user may have an interest in

#### Content-based Filtering
This approach utilizes a series of discrete characteristics of an item in order to recommend additional items with similar properties

#### Hybrid Recommender
This one combines the previous two approaches

#### Common Challenges
In my project, I will focus on building a collaborative filtering engine. In collaborative filtering, there are typically following challenges:
* cold start
* data sparsity
* popular bias (how to recommend products from the tail of product distribution)
* scalability (computation grows as number of users and items grow)
* pool relationship between like-minded yet sparse users

<p align="center">
  <a href="https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/movie_recommender/movie_recommendation_using_KNN.ipynb">
    <img alt="Long Tail Property in Ratings Distribution" title="Long Tail Property in Ratings Distribution" src="https://media.springernature.com/lw785/springer-static/image/chp%3A10.1007%2F978-3-319-29659-3_2/MediaObjects/334607_1_En_2_Fig1_HTML.gif">
  </a>
</p>

Above chart is the distribution of item rating frequency. This distribution often satisfies a property in real-world settings, which is referred to as [the long-tail property [1]](https://www.springer.com/cda/content/document/cda_downloaddocument/9783319296579-c1.pdf?SGWID=0-0-45-1554478-p179516130). According to this property, only a small fraction of the items are rated frequently. Such items are referred to as popular items. The vast majority of items are rated rarely. 

In most cases, high-frequency items tend to be relatively competitive items with little profit for the merchant. On the other hand, the lower frequency items have larger profit margins. However, many recommendation algorithms have a tendency to suggest popular items rather than infrequent items. This phenomenon also has a negative impact on diversity, and users may often become bored by receiving the same set of recommendations of popular items

#### Solution
Use matrix factorization technique to train model to learn user-item interaction by capturing user information in user latent factors and item information in item latent factors. Meanwhile, matrix factorization technique can significantly reduce dimensionality and sparsity and it will reduce huge amount of memory footprint and make our system more scalable


## Movie Recommender System Development
In this project, I focus on collaborative filtering recommender systems since they are widely used and well research in many different business and consistently provide good business values. It'd be very cool I can develop **Movie Recommender Systems** for myself. Let's see what movie recommendations my recommender offers me.

#### Datasets
I use [MovieLens Datasets [2]](https://grouplens.org/datasets/movielens/latest/). This dataset describes 5-star rating and free-text tagging activity from MovieLens, a movie recommendation service. It contains 27753444 ratings and 1108997 tag applications across 58098 movies. These data were created by 283228 users between January 09, 1995 and September 26, 2018. This dataset was generated on September 26, 2018.

#### Models
I start with basic and easy-implment models for my recommender system. As I want to improve my system's recommendations, I use more complex models. Eventually, I use neural networks for the recommender system. Following is the list of three models I'd like to use.
 - **KNN Item Based Collaborative Filtering**
 - **Alternating Least Square** (ALS) Matrix Factorization
 - **Neural Collaborative Filtering** Approach


### [Movie Recommendation Engine Development with KNN](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/movie_recommender/movie_recommendation_using_KNN.ipynb)
Collaborative filtering based systems use the actions of users to recommend other items. In general, they can either be user-based or item-based. Item-based approach is usually prefered than user-based approach. User-based approach is often harder to scale because of the dynamic nature of users, whereas items usually don't change much, so item-based approach often can be computed offline.

KNN is a perfect go-to model for this use case and KNN is a very good baseline for recommender system development. In item-based collaborative filtering, KNN will use a pre-defined distance metric to find clusters of similar items based on users' ratings, and make recommendations using the distance metric in item ratings of top-k [nearest neighbors [1]]((https://www.springer.com/cda/content/document/cda_downloaddocument/9783319296579-c1.pdf?SGWID=0-0-45-1554478-p179516130)).

#### Let's Make Some Recommendations
"Iron Man" is one of my favorite movies so I want to test what movie recommendations my system is giving me. It's very cool to see my recommender system give me recommendations

Check out detailed source code and instruction of commands (see the parse_args function) in [knn_recommender.py](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/movie_recommender/src/knn_recommender.py)

Run KNN recommender system:
```
python src/knn_recommender.py --movie_name "Iron Man" --top_n 10
```

Output:
```
You have input movie: Iron Man
Found possible matches in our database: ['Iron Man (2008)', 'Iron Man 3 (2013)', 'Iron Man 2 (2010)']

Recommendation system start to make inference ...
It took my system 1.38s to make inference
Recommendations for Iron Man:
1: Bourne Ultimatum, The (2007), with distance of 0.4221
2: Sherlock Holmes (2009), with distance of 0.4194
3: Inception (2010), with distance of 0.3934
4: Avatar (2009), with distance of 0.3836
5: WALL·E (2008), with distance of 0.3835
6: Star Trek (2009), with distance of 0.3753
7: Batman Begins (2005), with distance of 0.3703
8: Iron Man 2 (2010), with distance of 0.3703
9: Avengers, The (2012), with distance of 0.3581
10: Dark Knight, The (2008), with distance of 0.3013
```

It's interesting that the recommended movies are from the same time period as "Iron Man". These movies were as much popular as "Iron Man" at that time. This is exactly the downside of item-based collaborative filtering system. It always offers users the same set of very popular items. Users might get bored at same point without seeing diversity of recommendations


### [Movie Recommendation Engine Development with ALS in Apache Spark](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/movie_recommender/movie_recommendation_using_ALS.ipynb)
**Alternating Least Square (ALS)** is one of state-of-the-art **Matrix Factorization** models under the context of distributed computing.

Matrix Factorization is simply a mathematical operation for matrices. It is usually more effective in collaborative filtering, because it allows us to discover the latent (hidden) features underlying the interactions between users and items (movies).

Advantages of collaborative filtering using Matrix Factorization:
* No need to know about item content
* "Item cold-start" problem is avoided
* User interest may change over time
* Explainability

There are matrix factorization methods such as **Singular Value Decomposition (SVD)** and **Non-Negative Matrix Factorization (NMF)**. I chose [Alternating Least Square (ALS) implemented in Spark](https://spark.apache.org/docs/preview/ml-collaborative-filtering.html#collaborative-filtering) because it is a parallel algorithm designed for a large-scale collaborative filtering problems (such as the Netflix Prize). This method is doing a pretty good job at resolving scalability and sparseness of the user profiles, and it's simple and scales well to very large datasets.

The basic ideas behind ALS are:
* Factorize a big matrix into two small matrix (A = Users * Items)
* Use two loss functions for gradient descent
* Alternative gradient descent between Users and Items matrices back and forth

If you are interested in the math part behind ALS, please read [Large-scale Parallel Collaborative Filtering for the Netflix Prize [3]](https://endymecy.gitbooks.io/spark-ml-source-analysis/content/%E6%8E%A8%E8%8D%90/papers/Large-scale%20Parallel%20Collaborative%20Filtering%20the%20Netflix%20Prize.pdf)

Hyperparameter tuning in Alternating Least Square:
* maxIter: the maximum number of iterations to run (defaults to 10)
* rank: the number of latent factors in the model (defaults to 10)
* regParam: the regularization parameter in ALS (defaults to 1.0)

#### Let's Make Some Recommendations
I will pretend a user and input my favorite movie "Iron Man" again into this new recommender system. Let's see what movies it recommends to me. Hope they are not the same boring list of popular movies

Check out detailed source code and instruction of commands (see the parse_args function) in [als_recommender.py](https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/movie_recommender/src/als_recommender.py)

Run Alternating Least Square recommender system:
```
python src/als_recommender.py --movie_name "Iron Man" --top_n 10
```

Output:
```
Recommendation system start to make inference ...

You have input movie: Iron Man
Found possible matches in our database: ['Iron Man (2008)', 'Iron Man 2 (2010)', 'Invincible Iron Man, The (2007)', 'Iron Man: Rise Of Technovore (2013)', 'Iron Man 3 (2013)', 'Iron Man & Captain America: Heroes United (2014)', 'Iron Man & Hulk:Heroes United (2013)', 'Iron Man (1951)', 'Iron Man (1931)']

Recommendations for Iron Man:
1: Nine Deaths of the Ninja (1985), with rating of 6.5423
2: Pearl Jam: Immagine in Cornice - Live in Italy 2006 (2007), with rating of 6.4800
3: Presumed Guilty (Presunto culpable) (2008), with rating of 6.2537
4: Summer Heights High (2007), with rating of 6.1016
5: Dark Dungeons (2014), with rating of 6.0401
6: Hillary's America: The Secret History Of The Democratic Party (2016), with rating of 6.0361
7: Countdown (2004), with rating of 6.0336
8: Heroes Above All (2017), with rating of 6.0326
9: Stone Cold Steve Austin: The Bottom Line on the Most Popular Superstar of All Time (2011), with rating of 6.0006
10: WWE: Ladies and Gentlemen, My Name Is Paul Heyman (2014), with rating of 5.9800
```

This new list of Movies are completely different from the list of KNN recommender, which is very interesting. I have never watch any one of movies from this new list. This new recommender is able to offer less-known movies to user and offer a bit of element of suprise too. We can potentially blend this list of recommendations into the previous list from KNN recommender so that this hybrid recommender can offer both popular and less-known content to users. 
