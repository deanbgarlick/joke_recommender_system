from pyspark.sql import SparkSession, SQLContext


from joke_recommender import recommend


def main(spark, sqlContext):
    recommend.main(spark, sqlContext)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("Recommendation").getOrCreate()
    sqlContext = SQLContext(spark.sparkContext)
    main(spark, sqlContext)
    spark.stop()