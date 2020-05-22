from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import SparkSession


def load_als_model():
    alsModel = ALSModel.load('s3://aws-emr-resources-257018485161-us-east-1/alsModel')
    return alsModel


def main():

    ratingsDF = spark.read.load(
        "s3://aws-emr-resources-257018485161-us-east-1/ratings_3_als.parquet"
    )

    (training, test) = ratingsDF.randomSplit([0.8, 0.2])

    als = ALS(
        maxIter=5,
        regParam=0.01,
        userCol="userID",
        itemCol="jokeID",
        ratingCol="rating",
        coldStartStrategy="drop",
    )

    alsModel = als.fit(training)

    test = test.na.drop()

    predictions = alsModel.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    alsModel.write().overwrite().save("s3://aws-emr-resources-257018485161-us-east-1/alsModel")


if __name__ == "__main__":

    spark = SparkSession.builder.appName("fit_ALS_model").getOrCreate()
    main()
    spark.stop()
