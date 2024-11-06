from pyspark.sql import SparkSession
from Dataset import Dataset
from TextPreprocessor.py import TextPreprocessor
import math

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

    return tag_post_df, post_text_df_raw

def preprocess_text(post_text_df_raw):
    return TextPreprocessor(post_text_df_raw).preprocess_text()

def vectorize_text():
    return

def cosine_similarity(x, y):
    dot_product = sum(a * b for a, b in zip(x, y))
    norm_x = math.sqrt(sum(a * a for a in x))
    norm_y = math.sqrt(sum(b * b for b in y))
    return dot_product / (norm_x * norm_y)

if __name__ == "__main__":
    spark_session = SparkSession.builder.appName("532 Project").getOrCreate()
    
    # Build the dataest
    tag_post_df, post_text_df_raw = build_dataset()

    # Preprocess the text data 
    post_text_df = preprocess_text(post_text_df_raw)

    # Vectorize the text data - (TODO: @prathik)
    vec_text_df = vectorize_text(post_text_df)

    # Compute the Cosine Similarity - (TODO: @abhinav)
    cosine_similarity()

    spark_session.stop()
