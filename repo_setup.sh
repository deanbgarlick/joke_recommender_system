wget http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words -O stopwords
aws s3 cp stopwords s3://aws-emr-resources-257018485161-us-east-1/stopwords
