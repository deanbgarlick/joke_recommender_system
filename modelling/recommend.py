from .fit_als import load_als_model
from .fit_lda import load_lda_model


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


def recommend_based_on_rating(alsModel):
    pass


def recommend_based_on_category(ldaModel, lda_category):
    pass


def main(spark):

    ldaModel = load_lda_model(spark)
    alsModel = load_als_model()
    print_topics(ldaModel)

