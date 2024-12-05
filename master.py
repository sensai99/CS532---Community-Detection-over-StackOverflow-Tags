import argparse
from CommunityDetection import CommunityDetection
from TFIDF import TF_IDF
from pyspark.sql import SparkSession

def main(tag_name):
    spark_session = SparkSession.builder \
        .appName("TagGraph") \
        .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12") \
        .getOrCreate()   
    TF_IDF(spark_session).prepare_TFIDF_vectors()
    Community_Detection = CommunityDetection(spark_session)
    recommended_tags,time = Community_Detection.community_detection(tag_name)
    print(recommended_tags)
    spark_session.stop()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--tag_name',default='.net')
    args = parser.parse_args()
    main(args.tag_name)


    