from Dataset import Dataset
from TextPreprocessor import TextPreprocessor, clean_text
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
    # result.show(20)
    print('Number of communities',result.select("component").distinct().count())
    result = graph.labelPropagation(maxIter=5)
    # result.show(20)
    print('Number of communities', result.select("label").distinct().count())

def get_graph_vertices_edges(TFIDF_vectors_path,tag_count):
    vectors = np.load(TFIDF_vectors_path)
    # with open('tag_id_name.json', "r") as file:
    #     tag_name_map = json.load(file)
    tag_ids = list(vectors.keys())[:tag_count]
    tag_vectors = list(vectors.values())[:tag_count]
    vertices = [(tag_id,'a') for tag_id in tag_ids]
   
    similarities = cosine_similarity(tag_vectors,tag_vectors)
    edges = []

    for i in tqdm(range(len(tag_ids))):
        for j in range(len(tag_ids)):
            edges.append((tag_ids[i],tag_ids[j],float(similarities[i][j])))

    edges = list(filter(lambda x: x[2] >= 0.5,edges))
    filtered_vertices = set([])
    for edge in edges:
        filtered_vertices.add(edge[0])
        filtered_vertices.add(edge[1])

    return [(tag_id,'a') for tag_id in filtered_vertices],edges

def community_detection(spark_session,TFIDF_vectors_path,tag_count=2000):
    vertices,edges = get_graph_vertices_edges(TFIDF_vectors_path,tag_count)
    print(len(vertices))
    print(len(edges))
    run_community_detection(spark_session.createDataFrame(vertices,["id","name"]),spark_session.createDataFrame(edges,["src", "dst", "weight"]))

def inference_TFIDF(post,trained_vectorizer,vector_path):
    query_vector = trained_vectorizer.transform([post]).toarray()[0]
    vector_map = np.load(vector_path)
    with open('tag_id_name.json', "r") as file:
        tag_name_map = json.load(file)
   
    query_vector = query_vector.reshape(1, -1)

    similarities = {}
    for key, value in vector_map.items():
        similarity = cosine_similarity(query_vector, value.reshape(1, -1))[0][0]
        similarities[key] = similarity

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    return [tag_name_map[id] for (id,similarity) in sorted_similarities[:20]]

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

    return vectorizer_body

if __name__ == "__main__":

    spark_session = SparkSession.builder \
    .appName("GraphFramesExample") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()
    spark_session.sparkContext.setCheckpointDir("../spark-checkpoints")

    spark_conf = spark_session.sparkContext.getConf()

    # Print all configurations
    for item in spark_conf.getAll():
        print(item)

    print("spark.executor.memory:", spark_conf.get("spark.executor.memory", "default_value"))
    print("spark.driver.memory:", spark_conf.get("spark.driver.memory", "default_value"))

    exit()

    vectors_save_path = 'datasets/vectors.npz'
    # # Second parameter is the number of TF-IDF features
    # trained_vectorizer = prepare_TFIDF_vectors(spark_session,vectors_save_path,1000)
    # # sample inference - currently runs only for tags in training data
    # sample_post = clean_text("""Git - only push up the most recent commit to github,<p>On my local git repo I've got many commits which include 'secret' connection strings :-)</p> <p>I don't want this history on github when I push it there.</p> <p>Essentially I want to push everything I have but want to get rid of a whole lot of history.</p> <p>Perhaps I would be better running in a branch for all my dev then just merging back to master before committing... then the history for master will just be the commit I want.</p> <p>I've tried running rebase:</p> <pre>git rebase –i HEAD~3</pre> <p>That went back 3 commits and then I could delete a commit.</p> <p>However ran into auto cherry-pick failed and it got quite complex.</p> <p>Any thoughts greatly appreciated... no big deal to can the history and start again if this gets too hard :-)</p>""")
    # suggested_tags = inference_TFIDF(sample_post,trained_vectorizer,vectors_save_path)
    # print(suggested_tags)
    # Second parameter is the number of tags over which community detection is run.
    community_detection(spark_session,vectors_save_path,3000)

    spark_session.stop()
