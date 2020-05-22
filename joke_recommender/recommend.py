from pyspark.sql.types import StructField, StructType, IntegerType, StringType, ArrayType
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors

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


def process_joke_input(userID, ldaModel, alsModel):
    pass


def get_user_predicted_ratings(userID, alsModel, sqlContext):
    userRatings = sqlContext.sql("SELECT * FROM ratings WHERE userID = {userID}".format(userID=userID))
    predictedRatingsForUser = alsModel.transform(userRatings)
    return predictedRatingsForUser


def joke_similarity(jokeOneID, jokeTwoID, sqlContext):
    jokeOneTopics = sqlContext.sql("SELECT topicDistribution FROM jokes WHERE jokeID = {jokeOneID}".format(jokeOneID=jokeOneID))
    jokeTwoTopics = sqlContext.sql("SELECT topicDistribution FROM jokes WHERE jokeID = {jokeTwoID}".format(jokeTwoID=jokeTwoID))
    foo = jokeOneTopics - jokeTwoTopics
    foo.show()


@udf(IntegerType())
def find_max_in_column_vectors(x):
    return int(x.argmax())


def main(spark, sqlContext):

    ldaModel = load_lda_model(spark)
    register_remove_punctuation_udf(spark)
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

    ldaModel.transform(jokesDF).show()
    jokesTransformed = ldaModel.transform(jokesDF)
    jokesDistribution = jokesTransformed.select("jokeID", "topicDistribution")
    jokesDistribution.show()
    jokesDistribution.write.mode("overwrite").parquet(
        "s3://aws-emr-resources-257018485161-us-east-1/jokestransformed.parquet"
    )
    #foo = jokesDistribution.rdd.map(lambda x: find_max_in_column_vectors(x))
    #foo.toDF().show()

    ldaModel.transform(jokesDF).select("jokeID", find_max_in_column_vectors("topicDistribution").alias("dominantTopic")).show()
    #ldaModel.transform(jokesDF).rdd.map(lambda x: x.topicDistribution).show()
    userRatingsPredictions = get_user_predicted_ratings(32, alsModel, sqlContext)
    userRatingsPredictions.show()

    joke_similarity(101, 102, sqlContext)
