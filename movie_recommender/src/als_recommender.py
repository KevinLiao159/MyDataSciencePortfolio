import gc

# spark imports
from pyspark.sql import Row
from pyspark.sql.functions import col, lower, lit
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS


class Dataset:
    """
    data object make loading raw files easier
    """
    def __init__(self, spark_session, filepath):
        """
        spark dataset constructor
        """
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.filepath = filepath
        # build spark data object
        self.RDD = self.load_file_as_RDD(self.filepath)
        self.DF = self.load_file_as_DF(self.filepath)

    def load_file_as_RDD(self, filepath):
        ratings_RDD = self.sc.textFile(filepath)
        header = ratings_RDD.take(1)[0]
        return ratings_RDD \
            .filter(lambda line: line != header) \
            .map(lambda line: line.split(",")) \
            .map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))) # noqa

    def load_file_as_DF(self, filepath):
        ratings_RDD = self.load_file_as_rdd(filepath)
        ratingsRDD = ratings_RDD.map(lambda tokens: Row(
            userId=int(tokens[0]), movieId=int(tokens[1]), rating=float(tokens[2]))) # noqa
        return self.spark.createDataFrame(ratingsRDD)


def tune_ALS(model, train_data, validation_data, maxIter, regParams, ranks):
    """
    grid search function to select the best model based on RMSE of
    validation data

    Parameters
    ----------
    model: spark ML model, ALS

    train_data: spark DF with columns ['userId', 'movieId', 'rating']

    validation_data: spark DF with columns ['userId', 'movieId', 'rating']

    maxIter: int, max number of learning iterations

    regParams: list of float, one dimension of hyper-param tuning grid

    ranks: list of float, one dimension of hyper-param tuning grid

    Return
    ------
    The best fitted ALS model with lowest RMSE score on validation data
    """
    # initial
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in regParams:
            # get ALS model
            als = model.setMaxIter(maxIter).setRank(rank).setRegParam(reg)
            # train ALS model
            model = als.fit(train_data)
            # evaluate the model by computing the RMSE on the validation data
            predictions = model.transform(validation_data)
            evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="rating",
                                            predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            print('{} latent factors and regularization = {}: '
                  'validation RMSE is {}'.format(rank, reg, rmse))
            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_regularization = reg
                best_model = model
    print('\nThe best model has {} latent factors and '
          'regularization = {}'.format(best_rank, best_regularization))
    return best_model


class AlsRecommender:
    """
    This a collaborative filtering recommender with Alternating Least Square
    Matrix Factorization, which is implemented by Spark
    """
    def __init__(self, spark_session, path_movies, path_ratings):
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.moviesDF = self._load_file(path_movies)
        self.ratingsDF = self._load_file(path_ratings)
        self.model = ALS(
            userCol='userId',
            itemCol='movieId',
            ratingCol='rating',
            coldStartStrategy="drop")

    def _load_file(self, filepath):
        """
        load csv file into memory as spark DF
        """
        return self.spark.read.load(filepath, format='csv',
                                    header=True, inferSchema=True)

    def tune_model(self, maxIter, regParams, ranks, split_ratio=(6, 2, 2)):
        """
        Hyperparameter tuning for ALS model

        Parameters
        ----------
        maxIter: int, max number of learning iterations

        regParams: list of float, regularization parameter

        ranks: list of float, number of latent factors

        split_ratio: tuple, (train, validation, test)
        """
        # split data
        train, val, test = self.ratingsDF.randomSplit(split_ratio)
        # holdout tuning
        self.model = tune_ALS(self.model, train, val,
                              maxIter, regParams, ranks)
        # test model
        predictions = self.model.transform(test)
        evaluator = RegressionEvaluator(metricName="rmse",
                                        labelCol="rating",
                                        predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        print('The out-of-sample RMSE of the best tuned model is:', rmse)
        # clean up
        del train, val, test, predictions, evaluator
        gc.collect()

    def set_model_params(self, maxIter, regParam, rank):
        """
        set model params for pyspark.ml.recommendation.ALS

        Parameters
        ----------
        maxIter: int, max number of learning iterations

        regParams: float, regularization parameter

        ranks: float, number of latent factors
        """
        self.model = self.model \
            .setMaxIter(maxIter) \
            .setRank(rank) \
            .setRegParam(regParam)

    def _regex_matching(self, fav_movie):
        """
        return the closest matches via SQL regex.
        If no match found, return None

        Parameters
        ----------
        fav_movie: str, name of user input movie

        Return
        ------
        list of indices of the matching movies
        """
        matchesDF = self.moviesDF \
            .filter(
                lower(
                    col('title')
                ).like('%{}%'.format(fav_movie.lower()))
            ) \
            .select('movieId', 'title')
        if matchesDF.head(1).isEmpty:
            print('Oops! No match is found')
        else:
            movieIds = matchesDF.rdd.map(lambda r: r[0]).collect()
            titles = matchesDF.rdd.map(lambda r: r[1]).collect()
            print('Found possible matches in our database: '
                  '{0}\n'.format([x for x in titles]))
            return movieIds

    def _append_ratings(self, userId, movieIds):
        """
        append a user's movie ratings to ratingsDF

        Parameter
        ---------
        userId: int, userId of a user

        movieIds: int, movieIds of user's favorite movies
        """
        # create new user rdd
        user_rdd = self.sc.parallelize(
            [(userId, movieId, 5.0) for movieId in movieIds])
        # transform to user rows
        user_rows = user_rdd.map(
            lambda x: Row(
                userId=int(x[0]),
                movieId=int(x[1]),
                rating=float(x[2])
            )
        )
        # transform rows to spark DF
        userDF = self.spark.createDataFrame(user_rows)
        # append to ratingsDF
        self.ratingsDF = self.ratingsDF.union(userDF)

    def _create_inference_data(self, userId, movieIds):
        """
        create a user with all movies except ones were rated for inferencing
        """
        # filter movies
        return self.moviesDF \
            .filter(~col('movieId').isin(movieIds)) \
            .withColumn('userId', lit(int(userId))) \
            .select(['userId', 'movieId'])

    def _inference(self, model, data, fav_movie, n_recommendations):
        """
        """

    def make_recommendations(self, fav_movie, n_recommendations):
        """
        """