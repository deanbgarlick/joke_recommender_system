from pyspark.sql import SparkSession

from modelling import recommend

spark = SparkSession.builder.appName("Recommendation").getOrCreate()
recommend.main(spark)