import math
import os
from Dataset import Dataset
from TextPreprocessor import TextPreprocessor
from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, collect_list, concat_ws, lit,trim, col,udf
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, DoubleType
from pyspark.ml.linalg import SparseVector, VectorUDT, DenseVector
from pyspark.sql import functions as F
# os.environ["PYSPARK_PYTHON"] = r"C:\Users\DELL\anaconda3\envs\532project\python.exe"
# os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\DELL\anaconda3\envs\532project\python.exe"


def build_dataset():
    # Build the required dataframes (running locally)
    # Note: running with is_local as True without google cloud SDK setup might throw errors
    dataset = Dataset(spark_session, is_local = True)

    tag_post_df = dataset.build_tag_post_df()
    post_text_df_raw = dataset.build_post_text_df()

    print('Schema of tag_post_df:')
    # tag_post_df.printSchema()

    print('-' * 50)

    print('Schema of post_text_df_raw:')
    # post_text_df_raw.printSchema()

    print('Number of tags in the raw dataset: ', tag_post_df.count())
    print('Number of posts in the raw dataset: ', post_text_df_raw.count())

    return tag_post_df, post_text_df_raw

def preprocess_text(post_text_df_raw):
    return TextPreprocessor(post_text_df_raw).preprocess_text()

def vectorize_text(text_df):
    text_df = text_df.withColumn('title_words',split(text_df['title']," ")).drop('title')
    text_df = text_df.withColumn('body_words',split(text_df['body']," ")).drop('body')
    numF = 20
    hash_tf = HashingTF(inputCol="title_words", outputCol="raw_features",numFeatures=numF)
    featurized_data = hash_tf.transform(text_df)
    featurized_data = featurized_data.repartition(200)

    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")

    idf_model = idf.fit(featurized_data.select('raw_features'))

    tfidf_data = idf_model.transform(featurized_data)
    tfidf_data.show()
    return tfidf_data
    # text_df = text_df.withColumn("text", concat_ws(" ", col("title"), col("body")))
    # text_df = text_df.withColumn("text", split(col("text"), " "))
    # hash_tf = HashingTF(inputCol="title_words", outputCol="raw_features",numFeatures=20)
    # featurized_data = hash_tf.transform(text_df)
    # featurized_data.show()
    # idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    # idf_model = idf.fit(featurized_data.select('raw_features'))
    # tfidf_data = idf_model.transform(featurized_data)
    # # tfidf_data.select("post_id", "tfidf_features").show(truncate=False)
    # tfidf_data = tfidf_data.select("post_id", "tfidf_features")
    # tfidf_data.show()
   

@udf(ArrayType(DoubleType()))
def average_vectors(vectors):
    n = len(vectors)
    if n == 0:
        return DenseVector([]) 
    summed_vector = [0.0] * len(vectors[0])
    for vec in vectors:
        for sum_val, vec_val in zip(summed_vector, vec):
            print(sum_val,vec_val)
            if vec_val != "," :
                print(type(sum_val), type(float(vec_val)))
        summed_vector = [
            sum_val + float(vec_val) if vec_val != "," else sum_val
            for sum_val, vec_val in zip(summed_vector, vec)
        ]
    return [x / n for x in summed_vector]
    
def convert_vector(tfidf_tuple):
    print("HERE")
    print(tfidf_tuple)
    features, indices, scores = tfidf_tuple
    tfidf_vector = [0] * features
    for index in indices:
        tfidf_vector[index] = scores[index]
    return tfidf_vector

# @udf(returnType = ArrayType(DoubleType()))
# def average_vectors(post_ids):
#     logger.info('fdafdfa')
#     logger.info(post_ids)
#     retVal = []
#     post_ids = [34404321,5850685]
#     for post_id in post_ids:
#         retVal.append(convert_vector(tfidf_data.filter(tfidf_data['post_id'] == post_id)['tfidf_features']))
#     return sum(retVal)/len(post_ids)

def cosine_similarity(x, y):
    dot_product = sum(a * b for a, b in zip(x, y))
    norm_x = math.sqrt(sum(a * a for a in x))
    norm_y = math.sqrt(sum(b * b for b in y))
    return dot_product / (norm_x * norm_y)

if __name__ == "__main__":
    spark_session = SparkSession.builder.appName("532 Project") \
    .config("spark.executor.heartbeatInterval", "100s") \
    .config("spark.network.timeout", "120s") \
    .getOrCreate()
    print(f"PYSPARK_PYTHON: {os.environ.get('PYSPARK_PYTHON')}")
    print(f"PYSPARK_DRIVER_PYTHON: {os.environ.get('PYSPARK_DRIVER_PYTHON')}")
    # Build the dataest
    tag_post_df, post_text_df_raw = build_dataset()
    tag_post_df = tag_post_df.limit(10)
    tag_post_df = tag_post_df.withColumn(
    "post_ids_array",
    F.split(F.col("post_ids"), ",").cast("array<string>")
    )
    tag_post_df = tag_post_df.withColumn(
        "post_ids_array",
        F.expr("transform(post_ids_array, x -> cast(x as int))")
    )
    new_post_ids = "20711943,12252506,35946365,12748096"

    window_spec = Window.orderBy(F.lit(1)) 
    tag_post_df = tag_post_df.withColumn(
        "post_ids",
        F.when(F.row_number().over(window_spec) <= 10, new_post_ids)  
        .otherwise(F.col("post_ids"))
    )

    tag_post_df.show()
    # Preprocess the text data 
    post_text_df = preprocess_text(post_text_df_raw)
    post_text_df = post_text_df.filter(post_text_df["score"] >= 1)
    post_text_df = post_text_df.filter(
        (trim(col("title")).isNotNull()) & (trim(col("title")) != "") & 
        (trim(col("body")).isNotNull()) & (trim(col("body")) != "")
    )
    post_text_df = post_text_df.limit(10)
    # Vectorize the text data - (TODO: @prathik)
    vec_text_df = vectorize_text(post_text_df)
    tag_post_df.show()
    tfidf_avg = tag_post_df.withColumn('tf_idf_vector',average_vectors(tag_post_df['post_ids']))
    average_vectors_udf = F.udf(average_vectors, ArrayType(DoubleType()))
    tfidf_avg.show()
    tfidf_data = vec_text_df.withColumn(
        "dense_tfidf_features", 
        F.udf(lambda x: x.toArray().tolist(), ArrayType(DoubleType()))(F.col("tfidf_features"))
    )
    tfidf_data.show()
    tag_post_exploded_df = tag_post_df.withColumn(
        "exploded_post_id",
        F.explode(F.col("post_ids_array"))
    )
    
    tag_post_exploded_df.limit(10)
    tag_post_exploded_df.show()
    tfidf_data.show()
    joined_df = tag_post_exploded_df.join(
        tfidf_data.select("post_id", "dense_tfidf_features"), 
        on=tag_post_exploded_df.exploded_post_id == tfidf_data.post_id, 
        how="inner"
    )
    joined_df.show()
    # result_df = joined_df.groupBy("tag_id").agg(
    #     average_vectors_udf(F.collect_list("dense_tfidf_features")).alias("avg_tfidf_features")
    # )

    # result_df.show(truncate=False)
    
    # Compute the Cosine Similarity - (TODO: @abhinav)
    cosine_similarity()

    spark_session.stop()
