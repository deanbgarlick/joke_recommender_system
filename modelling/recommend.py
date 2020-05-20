from pyspark.sql import SparkSession

from fit_als import load_als_model
from fit_lda import load_lda_model


def main(spark):

    ldaModel = load_lda_model(spark)
    alsModel = load_als_model()


if __name__ == "__main__":

    spark = SparkSession.builder.appName("fit_LDA_model").getOrCreate()
    main(spark)
