from pyspark.ml import Pipeline, PipelineModel
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

    (training, test) = jokesDF.randomSplit([0.8, 0.2])

    stopwords = sc.textFile(
        "s3://aws-emr-resources-257018485161-us-east-1/stopwords"
    ).collect()

    numTopics = 20

    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    remover = StopWordsRemover(
        stopWords=stopwords, inputCol=tokenizer.getOutputCol(), outputCol="filtered"
    )
    vectorizer = CountVectorizer(
        inputCol=remover.getOutputCol(), outputCol="features", minDF=2
    )
    lda = LDA(k=numTopics)

    pipeline = Pipeline(stages=[tokenizer, remover, vectorizer, lda])

    model = pipeline.fit(training)
    model.write().overwrite().save("s3://aws-emr-resources-257018485161-us-east-1/ldaPipelineModel")

    prediction = model.transform(test)

    spark.stop()


def print_topic(topic, i):
    print("\n\n")
    print(str(topic) + ": " +str(i))
    topic.foreach(lambda term, weight: print(str(term) + ": " + str(weight)))


def print_topics():

    ldaPipelineModel = PipelineModel.load("s3://aws-emr-resources-257018485161-us-east-1/ldaPipelineModel")
    countVectorizer = ldaPipelineModel.stages[2]
    vocabList = countVectorizer.vocabulary
    ldaModel = ldaPipelineModel.stages[3]
    topicIndices = ldaModel.describeTopics(maxTermsPerTopic=5)
    topics = topicIndices.rdd.map(lambda terms, termWeights: terms.map(vocabList).zip(termWeights))
    topics.zipWithIndex().foreach(lambda topic, i: print_topic(topic, i))


if __name__ == "__main__":
    main()
    print_topics()
