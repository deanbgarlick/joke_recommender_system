from pyspark.ml.clustering import LDA, OnlineLDAOptimizer
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.sql import SparkSession
from pyspark.sql.context import SQLContext
from pyspark.sql.types import StructField, StructType, IntegerType, StringType


if __name__ == "__main__":

    spark = SparkSession.builder.appName("fit_LDA_model").getOrCreate()

    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

    jokesDF = spark.read.schema(StructType([ StructField("jokeID", IntegerType(), False), StructField("text", StringType(),False)])).csv('s3://aws-emr-resources-257018485161-us-east-1/jokes_3.csv', header='true')

    tokenizer = Tokenizer(inputCol="text", outputCol ="tokens")
    tokenizedDF = tokenizer.transform(jokesDF)

    stopwords = sc.textFile("s3://aws-emr-resources-257018485161-us-east-1/stopwords").collect()
    remover = StopWordsRemover(stopWords=stopwords, inputCol="tokens", outputCol="filtered")
    filteredDF = remover.transform(tokenizedDF)

    vectorizer = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5).fit(filteredDF)

    countVectors = vectorizer.transform(filteredDF).select("id", "features")

    lda_countVector = countVectors.map(lambda x: (x['id'], x['features']))

    numTopics = 20

    lda = LDA().setOptimizer(OnlineLDAOptimizer().setMiniBatchFraction(0.8)).setK(numTopics).setMaxIterations(3).setDocConcentration(-1).setTopicConcentration(-1)

    ldaModel = lda.run(lda_countVector)