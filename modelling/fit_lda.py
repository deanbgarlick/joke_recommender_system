from pyspark.ml.clustering import LDA
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, IntegerType, StringType


def main():

    spark = SparkSession.builder.appName("fit_LDA_model").getOrCreate()

    sc = spark.sparkContext

    jokesDF = spark.read.schema(
        StructType(
            [
                StructField("jokeID", IntegerType(), False),
                StructField("text", StringType(), False),
            ]
        )
    ).csv("s3://aws-emr-resources-257018485161-us-east-1/jokes_3.csv", header="true")

    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    tokenizedDF = tokenizer.transform(jokesDF)

    stopwords = sc.textFile(
        "s3://aws-emr-resources-257018485161-us-east-1/stopwords"
    ).collect()
    remover = StopWordsRemover(
        stopWords=stopwords, inputCol="tokens", outputCol="filtered"
    )
    filteredDF = remover.transform(tokenizedDF)

    vectorizer = CountVectorizer(
        inputCol="filtered", outputCol="features", minDF=2
    ).fit(filteredDF)

    countVectors = vectorizer.transform(filteredDF).select(["jokeID", "features"])

    numTopics = 20

    lda = LDA(k=numTopics)
    ldaModel = lda.fit(countVectors)
    ldaModel.save("s3://aws-emr-resources-257018485161-us-east-1/ldaModel")

    spark.stop()


if __name__ == "__main__":
    main()
