from pyspark.sql import SparkSession
from Dataset import Dataset

def build_dataset():
    # Build the required dataframes (running locally)
    # Note: running with is_local as True without google cloud SDK setup might throw errors
    dataset = Dataset(spark_session, is_local = True)

    tag_post_df = dataset.build_tag_post_df()
    post_text_df_raw = dataset.build_post_text_df()

    print('Schema of tag_post_df:')
    tag_post_df.printSchema()

    print('-' * 50)

    print('Schema of post_text_df_raw:')
    post_text_df_raw.printSchema()

    print('Number of tags in the raw dataset: ', tag_post_df.count())
    print('Number of posts in the raw dataset: ', post_text_df_raw.count())

def preprocess_text():
    return

def vectorize_text():
    return

def cosine_similarity():
    return

if __name__ == "__main__":
    spark_session = SparkSession.builder.appName("532 Project").getOrCreate()
    
    # Build the dataest
    build_dataset()

    # Preprocess the text data - (TODO: @durga)
    preprocess_text()

    # Vectorize the text data - (TODO: @prathik)
    vectorize_text()

    # Compute the Cosine Similarity - (TODO: @abhinav)
    cosine_similarity()

    spark_session.stop()