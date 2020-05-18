from pyspark.sql import SparkSession
from pyspark.sql.context import SQLContext
from pyspark.sql.functions import lit, countDistinct
from pyspark.sql.types import StructField, StructType, IntegerType, DoubleType


if __name__ == '__main__':

    spark = SparkSession \
        .builder \
        .appName("ALS_Data_Prep") \
        .getOrCreate()

    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

    inputDF = spark.read.csv('s3://aws-emr-resources-257018485161-us-east-1/ratings_3.csv', header='true', inferSchema='true').withColumnRenamed('_c0', 'userID').drop('0')

    fields = [StructField('userID', IntegerType(), True), StructField('jokeID', IntegerType(), True), StructField('rating', DoubleType(), True)]
    schema = StructType(fields)
    allJokeRatingsDF = sqlContext.createDataFrame(sc.emptyRDD(), schema)

    for jokeID in inputDF.columns:
        if jokeID != 'userID':
            if jokeID in ['105', '115', '125']:
                jokeRatings = inputDF.select(['userID', jokeID])
                distinctJokeRatings = jokeRatings.select(countDistinct(jokeID)).collect()[0][0]
                if distinctJokeRatings != 1:
                    jokeRatings = jokeRatings.withColumn('jokeID', lit(int(jokeID)))
                    jokeRatings = jokeRatings.withColumn("rating", jokeRatings[jokeID].cast(DoubleType()))
                    jokeRatings = jokeRatings.drop(jokeID)
                    allJokeRatingsDF = allJokeRatingsDF.union(jokeRatings)

    allJokeRatingsDF.write.parquet("s3://aws-emr-resources-257018485161-us-east-1/ratings_3_als.parquet")

    spark.stop()
