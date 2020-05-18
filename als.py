from pyspark.sql import SparkSession
from pyspark.sql.context import SQLContext


if __name__ == '__main__':

    spark = SparkSession \
        .builder \
        .appName("ALS_Data_Prep") \
        .getOrCreate()

    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

    ratingsDF = spark.read.load("s3://aws-emr-resources-257018485161-us-east-1/ratings_3_als.parquet")

    spark.stop()
