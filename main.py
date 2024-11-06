from pyspark.sql import SparkSession
from Dataset import Dataset
from TextPreprocessor.py import TextPreprocessor
from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, concat_ws, col
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

def vectorize_text(text_df):
    text_df = text_df.withColumn("text", concat_ws(" ", col("title"), col("body")))
    text_df = text_df.withColumn("text", split(col("text"), " "))
    
    hash_tf = HashingTF(inputCol="title_words", outputCol="raw_features",numFeatures=20)
    featurized_data = hash_tf.transform(text_df)
    
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    idf_model = idf.fit(featurized_data.select('raw_features'))
    tfidf_data = idf_model.transform(featurized_data)
    
    # tfidf_data.select("post_id", "tfidf_features").show(truncate=False)
    tfidf_data = tfidf_data.select("post_id", "tfidf_features")
    return tfidf_data
    
def convert_vector(tfidf_tuple):
    print(tfidf_tuple)
    features, indices, scores = tfidf_tuple
    tfidf_vector = [0] * features
    for index in indices:
        tfidf_vector[index] = scores[index]
    return tfidf_vector

@udf(returnType = ArrayType(DoubleType()))
def average_vectors(post_ids):
    logger.info('fdafdfa')
    logger.info(post_ids)
    retVal = []
    post_ids = [34404321,5850685]
    for post_id in post_ids:
        retVal.append(convert_vector(tfidf_data.filter(tfidf_data['post_id'] == post_id)['tfidf_features']))
    return sum(retVal)/len(post_ids)

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
    tfidf_avg = tag_post_df.withColumn('tf_idf_vector',average_vectors(tag_post_df['postIds']))
    
    # Compute the Cosine Similarity - (TODO: @abhinav)
    cosine_similarity()

    spark_session.stop()
