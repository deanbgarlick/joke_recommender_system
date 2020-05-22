from pyspark.sql import SparkSession

from modelling import recommend


def main():
    spark = SparkSession.builder.appName("Recommendation").getOrCreate()
    recommend.main(spark)


if __name__ == "__main__":
    main()