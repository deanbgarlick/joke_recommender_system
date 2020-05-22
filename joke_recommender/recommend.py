from pyspark.sql.types import StructField, StructType, IntegerType, StringType, ArrayType
from pyspark.sql.functions import udf

from .fit_als import load_als_model
from .fit_lda import load_lda_model, register_remove_punctuation_udf


def print_topic(topic, i):
    print("\n\n")
    print("Topic: " + str(i))
    list(map(lambda x: print(str(x[0]) + ": " + str(x[1])), topic)) # x = ( term, weight)


def print_topics(ldaPipelineModel):
    countVectorizer = ldaPipelineModel.stages[3]
    vocabList = countVectorizer.vocabulary
    ldaModel = ldaPipelineModel.stages[4]
    topicIndices = ldaModel.describeTopics(maxTermsPerTopic=5)
    topics = topicIndices.rdd.map(lambda x: list(zip(list(map(lambda n: vocabList[n], x.termIndices)), x.termWeights)))
    topicsWithIndex = topics.zipWithIndex()
    list(map(lambda x: print_topic(x[0], x[1]), topicsWithIndex.collect()))


def process_joke_input(userId, ldaModel, alsModel):
    pass

def recommend_based_on_rating(alsModel, numRecommend):
    pass


def recommend_based_on_category(ldaModel, alsModel, ldaCategory, numRecommend):
    pass


def main(spark):

    ldaModel = load_lda_model(spark)
    register_remove_punctuation_udf()
    alsModel = load_als_model()
    print_topics(ldaModel)

    ratingsDF = spark.read.load(
        "s3://aws-emr-resources-257018485161-us-east-1/ratings_3_als.parquet"
    )
    ratingsDF.createOrReplaceTempView("ratings")

    jokesDF = spark.read.schema(
        StructType(
            [
                StructField("jokeID", IntegerType(), False),
                StructField("raw_text", StringType(), False),
            ]
        )
    ).csv("s3://aws-emr-resources-257018485161-us-east-1/jokes_3.csv", header="true")
    jokesDF.createOrReplaceTempView("jokes")

    find_max_in_column_vectors = udf(lambda x: x.toDense.values.toSeq.indices.maxBy(x.toDense.values), IntegerType())
    ldaModel.transform(jokesDF).select(find_max_in_column_vectors("topicDistribution").alias("dominantTopic")).show()