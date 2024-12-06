from Dataset import Dataset
from TextPreprocessor import TextPreprocessor, clean_text
from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, collect_list, concat_ws, lit,trim, col,udf
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, DoubleType, StringType
from pyspark.ml.linalg import SparseVector, VectorUDT, DenseVector
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import desc
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from graphframes import GraphFrame
import community as community_louvain
from networkx.algorithms.community import label_propagation_communities
import networkx as nx

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

def run_community_detection(vertices,edges,py=True):
    if py==True:
        python_graph = nx.Graph()
        vertices_pd = vertices.toPandas()
        edges_pd = edges.toPandas()
        print(vertices_pd.columns)
        for _, row in vertices_pd.iterrows():
            if 'attribute' in row:
                python_graph.add_node(row['id'], attribute=row['attribute'])
            else:
                python_graph.add_node(row['id'])  # Handle case without 'attribute'
    
        for _, row in edges_pd.iterrows():
            python_graph.add_edge(row['src'], row['dst'], weight=row['weight'])
    
        print("Inside NetworkX Creation")
        communities = label_propagation_communities(python_graph)
        node_labels = {node: idx for idx, community in enumerate(communities) for node in community}
        result = pd.DataFrame(list(node_labels.items()), columns=["id", "label"])
        print(result)
        distinct_count = result['label'].nunique()
        print(f"Number of distinct communities: {distinct_count}")
    else:
        graph = GraphFrame(vertices, edges)
        result = graph.connectedComponents()
        print('Number of communities',result.select("component").distinct().count())
        result = graph.labelPropagation(maxIter=5)
        result.show(20)   
        print('Number of communities', result.select("label").distinct().count())

    print('After NetworkX Creation')
    return result

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
        for j in range(i+1,len(tag_ids)):
            edges.append((tag_ids[i],tag_ids[j],float(similarities[i][j])))

    edges = list(filter(lambda x: x[2] >= 0.5,edges))
    filtered_vertices = set([])
    for edge in edges:
        filtered_vertices.add(edge[0])
        filtered_vertices.add(edge[1])

    return [(tag_id,tag_name_map[tag_id]) for tag_id in filtered_vertices],edges

def inference_TFIDF_tag(tag_name,vector_path):
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
    for idx, (id, similarity) in enumerate(sorted_similarities[:10], start=1):
        print(f"Rank {idx}: tag = {tag_name_map[id]}, Similarity = {similarity:.4f}")
    return sorted_similarities
    

def inference_TFIDF_post(post,trained_vectorizer,vector_path):
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

def inference_graph(tag_name,vertices_df,edges_df,community_df):
    tag_id = vertices_df.filter(vertices_df.name == tag_name).first()['id']
    community_id = community_df.filter(community_df.id == tag_id).first()['label']
    community_df = community_df.filter(community_df.label == community_id)

    tag_id_edges = edges_df.filter((edges_df.src == tag_id) | (edges_df.dst == tag_id))
    target_vertex = udf(lambda src, dest: src if dest==tag_id else dest, StringType())
    tag_id_edges = tag_id_edges.withColumn('id',target_vertex(tag_id_edges.src,tag_id_edges.dst))

    search_space_df = tag_id_edges.join(community_df, on="id", how="inner")
    search_space_df = search_space_df.sort(desc("weight")).limit(10)
    search_space_df.show()
    suggested_tag_rows = search_space_df.head(10)
    
    return [row["name"] for row in suggested_tag_rows]

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
    try:
        spark_session = SparkSession.builder \
        .appName("GraphFramesExample") \
        .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
        .getOrCreate()
        spark_session.sparkContext.setCheckpointDir("../spark-checkpoints")

        vectors_save_path = 'datasets/vectors.npz'
        # # Second parameter is the number of TF-IDF features
        # trained_vectorizer = prepare_TFIDF_vectors(spark_session,vectors_save_path,1000)
        # # sample inference - currently runs only for tags in training data
        # sample_post = clean_text("""Git - only push up the most recent commit to github,<p>On my local git repo I've got many commits which include 'secret' connection strings :-)</p> <p>I don't want this history on github when I push it there.</p> <p>Essentially I want to push everything I have but want to get rid of a whole lot of history.</p> <p>Perhaps I would be better running in a branch for all my dev then just merging back to master before committing... then the history for master will just be the commit I want.</p> <p>I've tried running rebase:</p> <pre>git rebase –i HEAD~3</pre> <p>That went back 3 commits and then I could delete a commit.</p> <p>However ran into auto cherry-pick failed and it got quite complex.</p> <p>Any thoughts greatly appreciated... no big deal to can the history and start again if this gets too hard :-)</p>""")
        # suggested_tags = inference_TFIDF_post(sample_post,trained_vectorizer,vectors_save_path)
        # print(suggested_tags)
        # # Second parameter is the number of tags over which community detection is run.
        # # graph = community_detection(spark_session,vectors_save_path,200)
        # inference_TFIDF_tag('android-xml',vectors_save_path)
        vertices,edges = get_graph_vertices_edges(vectors_save_path,9000)
        print(len(vertices))
        print(len(edges))
        vertices_df = spark_session.createDataFrame(vertices,["id","name"])
        edges_df = spark_session.createDataFrame(edges,["src", "dst", "weight"])
        community_df = run_community_detection(vertices_df,edges_df,False)
        suggested_tags = inference_graph('android-xml',vertices_df,edges_df,community_df)
        print(suggested_tags)
    except():
        spark_session.stop()
