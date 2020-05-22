from pyspark.sql import SparkSession
from pyspark.sql.context import SQLContext
from pyspark.sql.functions import lit, countDistinct, when
from pyspark.sql.types import StructField, StructType, IntegerType, DoubleType


def main(spark, sqlContext):

    sc = spark.sparkContext

    inputDF = (
        spark.read.csv(
            "s3://aws-emr-resources-257018485161-us-east-1/ratings_3.csv",
            header="true",
            inferSchema="true",
        )
            .withColumnRenamed("_c0", "userID")
            .drop("0")
    )

    fields = [
        StructField("userID", IntegerType(), True),
        StructField("jokeID", IntegerType(), True),
        StructField("rating", DoubleType(), True),
    ]
    schema = StructType(fields)
    allJokeRatingsDF = sqlContext.createDataFrame(sc.emptyRDD(), schema)

    for jokeID in inputDF.columns:
        if jokeID != "userID":
           if jokeID != "0":
                jokeRatings = inputDF.select(["userID", jokeID])
                jokeRatings = jokeRatings.withColumn("jokeID", lit(int(jokeID)))
                jokeRatings = jokeRatings.withColumn(
                    "rating", when(jokeRatings.jokeID.isin([99.0,99.,99], None)).otherwise(jokeRatings.jokeID.cast(DoubleType()))
                )
                jokeRatings = jokeRatings.drop(jokeID)
                allJokeRatingsDF = allJokeRatingsDF.union(jokeRatings)

    allJokeRatingsDF.write.mode("overwrite").parquet(
        "s3://aws-emr-resources-257018485161-us-east-1/ratings_3_als.parquet"
    )


if __name__ == "__main__":

    spark = SparkSession.builder.appName("prep_data_for_ALS").getOrCreate()
    sc = spark.sparkContext
    sqlContext = SQLContext(sc)
    main(spark, sqlContext)
    spark.stop()
