from pyspark.sql import SparkSession

from joke_recommender import recommend


def main(spark):
    recommend.main(spark)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("Recommendation").getOrCreate()
    main(spark)
    spark.stop()