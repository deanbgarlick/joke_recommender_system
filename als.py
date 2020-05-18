from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.context import SQLContext


if __name__ == "__main__":

    spark = SparkSession.builder.appName("ALS_Data_Prep").getOrCreate()

    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

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

    model = als.fit(training)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    spark.stop()
