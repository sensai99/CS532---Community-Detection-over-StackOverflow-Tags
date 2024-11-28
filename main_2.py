from Dataset import Dataset
from TextPreprocessor import TextPreprocessor
from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, collect_list, concat_ws, lit,trim, col,udf
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, DoubleType
from pyspark.ml.linalg import SparseVector, VectorUDT, DenseVector
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from graphframes import GraphFrame


def build_dataset(spark_session):
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

def run_community_detection(vertices,edges):
    graph = GraphFrame(vertices, edges)
    result = graph.connectedComponents()
    # result.show()
    print('Number of communities',result.select("component").distinct().count())
    result = graph.labelPropagation(maxIter=5)
    # result.show()   
    print('Number of communities', result.select("label").distinct().count())

def get_graph_vertices_edges(TFIDF_vectors_path,tag_count):
    vectors = np.load(TFIDF_vectors_path)
    with open('tag_id_name.json', "r") as file:
        tag_name_map = json.load(file)
    tag_ids = list(vectors.keys())[:tag_count]
    tag_vectors = list(vectors.values())[:tag_count]
    vertices = [(tag_id,tag_name_map[tag_id]) for tag_id in tag_ids]
   
    similarities = cosine_similarity(tag_vectors,tag_vectors)
    edges = []

    for i in tqdm(range(len(tag_ids))):
        for j in range(len(tag_ids)):
            edges.append((tag_ids[i],tag_ids[j],float(similarities[i][j])))

    return vertices,edges

def community_detection(spark_session,TFIDF_vectors_path,tag_count=2000):
    vertices,edges = get_graph_vertices_edges(TFIDF_vectors_path,tag_count)
    run_community_detection(spark_session.createDataFrame(vertices,["id"]),spark_session.createDataFrame(edges,["src", "dst", "weight"]))

def inference_TFIDF(tag_name,vector_path):
    vector_map = np.load(vector_path)
    with open('tag_name_id.json', "r") as file:
        tag_id_map = json.load(file)
    with open('tag_id_name.json', "r") as file:
        tag_name_map = json.load(file)
   
    id = tag_id_map[tag_name]
    query_vector = vector_map[id].reshape(1, -1)

    similarities = {}
    for key, value in vector_map.items():
        similarity = cosine_similarity(query_vector, value.reshape(1, -1))[0][0]
        similarities[key] = similarity

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    for idx, (id, similarity) in enumerate(sorted_similarities[:20], start=1):
        print(f"Rank {idx}: tag = {tag_name_map[id]}, Similarity = {similarity:.4f}")
    return sorted_similarities

def prepare_TFIDF_vectors(spark_session,save_path,feature_size=1000):
 
    # Build the dataset
    tag_post_df, post_text_df_raw = build_dataset(spark_session)

    # preprocess tag data
    tag_post_df = tag_post_df.withColumn(
    "post_ids_array",
    F.split(F.col("post_ids"), ",").cast("array<string>")
    ).drop('post_ids')

    # Preprocess the post text data 
    post_text_df = preprocess_text(post_text_df_raw)
    post_text_df = post_text_df.filter(post_text_df["score"] >= 1)
    post_text_df = post_text_df.filter(
        (trim(col("title")).isNotNull()) & (trim(col("title")) != "") & 
        (trim(col("body")).isNotNull()) & (trim(col("body")) != "")
    )

    # convert to pandas dataframe
    tag_post_df_pandas = tag_post_df.toPandas()
    post_text_df_pandas = post_text_df.toPandas()

    tag_ids = []
    tag_names = []
    tag_post_ids = []

    for index,row in tag_post_df_pandas.iterrows():
        tag_ids.append(row['tag_id'])
        tag_names.append(row['tag_name'])
        tag_post_ids.append(row['post_ids_array'])

    body_texts = []
    post_ids = []
    for index,row in post_text_df_pandas.iterrows():
        body_texts.append(row['body'])
        post_ids.append(row['post_id'])


    tag_name_map = {str(tag_id):tag_name for tag_id,tag_name in zip(tag_ids,tag_names)}
    tag_id_map = {tag_name:str(tag_id) for tag_id,tag_name in zip(tag_ids,tag_names)}
    with open('tag_name_id.json', 'w') as file:
        json.dump(tag_id_map,file)
    with open('tag_id_name.json', 'w') as file:
        json.dump(tag_name_map,file)

    # Train TF-IDF vectorizer (1000 features)
    vectorizer_body = TfidfVectorizer(
    analyzer = 'word', 
    strip_accents = None, 
    encoding = 'utf-8', 
    preprocessor=None, 
    max_features=feature_size)

    vectors = vectorizer_body.fit_transform(body_texts).toarray()
    # post id to vector map
    post_id_vector_map = {post_id: vectors[index] for index,post_id in enumerate(post_ids)}

    # tag id to vector map
    tag_vector_map = {}
    for index,tag_id in enumerate(tag_ids):
        tag_posts = tag_post_ids[index]
        tag_posts_vectors = []
        for post_id in tag_posts:
            if post_id in post_ids: tag_posts_vectors.append(post_id_vector_map[post_id])
        
        tag_vector_map[str(tag_id)] = np.mean(np.array(tag_posts_vectors),axis=0) if len(tag_posts_vectors)>0 else np.zeros(feature_size)

    np.savez(save_path, **tag_vector_map)

if __name__ == "__main__":

    spark_session = SparkSession.builder \
    .appName("GraphFramesExample") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()
    spark_session.sparkContext.setCheckpointDir("../spark-checkpoints")

    vectors_save_path = 'vectors.npz'
    # Second parameter is the number of TF-IDF features
    prepare_TFIDF_vectors(spark_session,vectors_save_path,1000)
    # sample inference - currently runs only for tags in training data
    inference_TFIDF('github',vectors_save_path)
    # Second parameter is the number of tags over which community detection is run.
    community_detection(spark_session,vectors_save_path,20)

    spark_session.stop()
