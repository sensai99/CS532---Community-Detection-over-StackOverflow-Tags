import pandas as pd
from pyspark.sql.types import FloatType, ArrayType
from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import broadcast, col, udf, desc
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType, StringType
import numpy as np
import json
import time
from utils import cosine_similarity

class CommunityDetection:
    def __init__(self,spark_session,tag_data_path = "datasets/tag_id_name.json",TFIDF_vector_path="datasets/vectors.npz"):
        self.spark_session = spark_session
        self.tag_data_path = tag_data_path
        self.TFIDF_vector_path = TFIDF_vector_path

    def load_tag_data(self):
        tag_df = pd.read_json(self.tag_data_path)
        return self.spark_session.createDataFrame(tag_df).withColumnRenamed("id", "tag_id").withColumnRenamed("name", "tag_name")

    def load_tfidf_data(self,tag_count):
        npz_data = np.load(self.TFIDF_vector_path, allow_pickle=True)
        ids = list(npz_data.keys())[:tag_count]
        vectors = list(npz_data.values())[:tag_count]
        vectors = [vec.tolist() for vec in vectors]
        schema = StructType([
            StructField("tag_id", StringType(), True),
            StructField("tfidf_vector", ArrayType(FloatType()), True)
        ])
        return self.spark_session.createDataFrame(zip(ids, vectors), schema=schema)

    def create_similarity_df(self,tfidf_df,similarity_threshold):
        cosine_similarity_udf = udf(cosine_similarity, FloatType())
        return tfidf_df.alias("a") \
            .join(tfidf_df.alias("b"), col("a.tag_id") < col("b.tag_id")) \
            .withColumn("cosine_similarity", cosine_similarity_udf(col("a.tfidf_vector"), col("b.tfidf_vector"))) \
            .filter(col("cosine_similarity") > similarity_threshold)
    
    def create_edges_df(self,similarity_df):
        return similarity_df.select(
            col("a.tag_id").alias("src"),
            col("b.tag_id").alias("dst"),
            col("cosine_similarity").alias("weight")
        )

    def create_vertices_df(self,tag_df, tfidf_df):
        return tag_df.join(tfidf_df, on="tag_id") \
            .select("tag_id", "tag_name") \
            .withColumnRenamed("tag_id", "id") \
            .withColumnRenamed("tag_name", "name")

    def perform_community_detection(self,graph,training_iterations):
        start_time = time.time()
        connected_components = graph.labelPropagation(maxIter=training_iterations)
        end_time = time.time()
        return connected_components, end_time - start_time
    
    def recommended_tags(self,tag_name, vertices_df,edges_df,community_df):
        tag_id = vertices_df.filter(vertices_df.name == tag_name).first()['id']
        community_id = community_df.filter(community_df.id == tag_id).first()['label']
        community_df = community_df.filter(community_df.label == community_id)

        tag_id_edges = edges_df.filter((edges_df.src == tag_id) | (edges_df.dst == tag_id))
        target_vertex = udf(lambda src, dest: src if str(dest)==str(tag_id) else dest, StringType())
        tag_id_edges = tag_id_edges.withColumn('id',target_vertex(tag_id_edges.src,tag_id_edges.dst))

        search_space_df = tag_id_edges.join(community_df, on="id", how="inner")
        search_space_df = search_space_df.sort(desc("weight")).limit(10)
        search_space_df.show()
        suggested_tag_rows = search_space_df.head(10)
        
        return [row["name"] for row in suggested_tag_rows]

    def community_detection(self,tag_name,tag_count=20,similarity_threshold=0.5,training_iterations=5):
        tag_df = self.load_tag_data()
        tfidf_df = self.load_tfidf_data(tag_count)

        similarity_df = self.create_similarity_df(tfidf_df,similarity_threshold)
        edges_df = self.create_edges_df(similarity_df)
        vertices_df = self.create_vertices_df(broadcast(tag_df), tfidf_df)

        graph = GraphFrame(vertices_df, edges_df)
        community_df,time_taken = self.perform_community_detection(graph,training_iterations)

        suggested_tags = self.recommended_tags(tag_name,vertices_df,edges_df,community_df)
        return suggested_tags , time_taken
