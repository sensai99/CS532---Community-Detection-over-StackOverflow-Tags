import argparse
from CommunityDetection import CommunityDetection
from TFIDF import TF_IDF
from Testing import evaluate
from pyspark.sql import SparkSession

def main(tag_name):
    # Initialize a Spark session with specific configuration for using graphframes
    spark_session = SparkSession.builder \
        .appName("TagGraph") \
        .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12") \
        .getOrCreate()   
    # Create a TF_IDF object and prepare TF-IDF vectors
    TF_IDF(spark_session).prepare_TFIDF_vectors()
    # Initialize CommunityDetection with the current Spark session
    Community_Detection = CommunityDetection(spark_session)
    # Perform community detection for the specified tag_name and print results
    recommended_tags,time = Community_Detection.community_detection(tag_name)
    print(recommended_tags)

    evaluate(Community_Detection)    
    # Stop the Spark session to free up resources
    spark_session.stop()
    
# Call main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--tag_name',default='.net')
    args = parser.parse_args()
    main(args.tag_name)


    