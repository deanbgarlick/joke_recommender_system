from pyspark.sql import SparkSession

from fit_als import load_als_model
from fit_lda import load_lda_model


def main():

    ldaModel = load_lda_model()
    alsModel = load_als_model()


if __name__ == "__main__":
    
    global spark 
    spark = SparkSession.builder.appName("fit_LDA_model").getOrCreate()
    
    global sc
    sc = spark.sparkContext
    main()
