from pyspark.sql import Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS


class Dataset:
    """
    data object make loading raw file easier and training spark.ml.ALS easier
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
        # build train, val, test
        self.train, self.validation, self.test = \
            self.split_data(self.DF, [6, 2, 2], seed=99)

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
            userId=int(tokens[0]), itemId=int(tokens[1]), rating=float(tokens[2]))) # noqa

        return self.spark.createDataFrame(ratingsRDD)

    def split_data(self, data, weights, seed):
        """
        Parameters
        ----------
        data: spark data object, RDD or Spark DF
        weights: list of float, eg. [8, 2]
        seed: random seed
        """
        return data.randomSplit(weights, seed)


def train_ALS(train_data, validation_data, maxIter, regParams, ranks):
    """
    grid search function to select the best model based on RMSE of
    validation data

    Parameters
    ----------
    train_data: spark DF with columns ['userId', 'itemId', 'rating']

    validation_data: spark DF with columns ['userId', 'itemId', 'rating']

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
            als = ALS(rank=rank, maxIter=maxIter, regParam=reg,
                      userCol='userId', itemCol='itemId', ratingCol='rating',
                      coldStartStrategy='drop', seed=99)
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
