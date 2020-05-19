from pyspark.ml.clustering import LocalLDAModel
from pyspark.ml.recommendation import ALSModel


def main():

    ldaModel = LocalLDAModel.load("s3://aws-emr-resources-257018485161-us-east-1/ldaModel")
    alsModel = ALSModel.load("s3://aws-emr-resources-257018485161-us-east-1/alsModel")

    num_topics=10
    topn_words=5
    topics = ldaModel.topicsMatrix().toArray()
    for topic in range(num_topics):
        print("\n\nTopic " + str(topic) + ":")
        for word in range(0, topn_words):
            print(" " + str(topics[word][topic]))