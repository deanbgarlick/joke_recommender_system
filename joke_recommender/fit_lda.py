from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, SQLTransformer
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, ArrayType


def remove_punctuation(text):
    text = text.lower()
    items_to_remove = ["'", '(', ')', '"', ':', ';', '!', '.', ',', '-', '_', '?']
    for item in items_to_remove:
        text = text.replace(item, '')
    print(text)
    return text


def register_remove_punctuation_udf(spark):
    spark.udf.register(
        "remove_punctuation_udf",
        lambda row: remove_punctuation(row),
        StringType()
    )


def load_lda_model(spark):
    register_remove_punctuation_udf(spark)
    ldaPipelineModel = PipelineModel.load("s3://aws-emr-resources-257018485161-us-east-1/ldaPipelineModel")
    #ldaPipelineModel.stages[0] = SQLTransformer(statement="SELECT jokeID, clean_text_udf(raw_text) text FROM __THIS__")
    return ldaPipelineModel


def test_clean_text(spark):

    jokesDF = spark.read.schema(
        StructType(
            [
                StructField("jokeID", IntegerType(), False),
                StructField("raw_text", StringType(), False),
            ]
        )
    ).csv("s3://aws-emr-resources-257018485161-us-east-1/jokes_3.csv", header="true")

    # jokesDF = jokesDF.withColumn("text", clean_text_udf("raw_text"))

    (training, test) = jokesDF.randomSplit([0.8, 0.2])

    register_remove_punctuation_udf(spark)

    pipeline = Pipeline(
        stages=[SQLTransformer(statement="SELECT jokeID, remove_punctuation_udf(raw_text) text FROM __THIS__")]
    )
    model=pipeline.fit(training)
    model.transform(test).show()


def main(spark, numTopics):

    jokesDF = spark.read.schema(
        StructType(
            [
                StructField("jokeID", IntegerType(), False),
                StructField("raw_text", StringType(), False),
            ]
        )
    ).csv("s3://aws-emr-resources-257018485161-us-east-1/jokes_3.csv", header="true")

    #jokesDF = jokesDF.withColumn("text", clean_text_udf("raw_text"))

    (training, test) = jokesDF.randomSplit([0.8, 0.2])

    register_remove_punctuation_udf(spark)

    stopwords = spark.sparkContext.textFile(
        "s3://aws-emr-resources-257018485161-us-east-1/stopwords"
    ).collect()

    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    remover = StopWordsRemover(
        stopWords=stopwords, inputCol=tokenizer.getOutputCol(), outputCol="filtered"
    )
    vectorizer = CountVectorizer(
        inputCol=remover.getOutputCol(), outputCol="features", minDF=2
    )
    lda = LDA(k=numTopics)

    pipeline = Pipeline(stages=[
        SQLTransformer(statement="SELECT jokeID, remove_punctuation_udf(raw_text) text FROM __THIS__"),
        tokenizer,
        remover,
        vectorizer,
        lda])

    model = pipeline.fit(training)
    model.write().overwrite().save("s3://aws-emr-resources-257018485161-us-east-1/ldaPipelineModel")

    prediction = model.transform(test)

    prediction.show()


if __name__ == "__main__":

    spark = SparkSession.builder.appName("fit_LDA_model").getOrCreate()
    test_clean_text(spark)
    main(spark, numTopics=4)
    spark.stop()
