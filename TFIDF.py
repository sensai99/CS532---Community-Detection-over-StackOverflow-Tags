from Dataset import Dataset
from TextPreprocessor import TextPreprocessor
from pyspark.sql import functions as F
from pyspark.sql.functions import trim, col
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import json


class TF_IDF:
    def __init__(self,session):
        self.spark_session = session
        
    def build_dataset(self):
        dataset = Dataset(self.spark_session, is_local = True)
        tag_post_df = dataset.build_tag_post_df()
        post_text_df_raw = dataset.build_post_text_df()
        return tag_post_df, post_text_df_raw

    def preprocess_text(self,post_text_df_raw):
        return TextPreprocessor(post_text_df_raw).preprocess_text()
    
    def prepare_TFIDF_vectors(self,vector_save_path = "datasets/vectors.npz",tag_save_path="datasets/tag_id_name.json",feature_size=1000):

        # Build the dataset
        tag_post_df, post_text_df_raw = self.build_dataset()

        # preprocess tag data
        tag_post_df = tag_post_df.withColumn(
        "post_ids_array",
        F.split(F.col("post_ids"), ",").cast("array<string>")
        ).drop('post_ids')

        # Preprocess the post text data 
        post_text_df = self.preprocess_text(post_text_df_raw)
        post_text_df = post_text_df.filter(post_text_df["score"] >= 1)
        post_text_df = post_text_df.filter(
            (trim(col("title")).isNotNull()) & (trim(col("title")) != "") & 
            (trim(col("body")).isNotNull()) & (trim(col("body")) != "")
        )

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

   
        tag_name_map = [{'tag_name':tag_name,'tag_id':str(tag_id)} for tag_id,tag_name in zip(tag_ids,tag_names)]
        with open(tag_save_path, 'w') as file:
            json.dump(tag_name_map,file)

        # Train TF-IDF vectorizer
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

        if vector_save_path: np.savez(vector_save_path, **tag_vector_map)

        return vectorizer_body