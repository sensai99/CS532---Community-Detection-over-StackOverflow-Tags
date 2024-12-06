from pyspark.sql.functions import explode, split, collect_list, concat_ws, lit

class Dataset:
    def __init__(self, spark_session, is_local = True, limit = None):
        self.spark_session = spark_session  # Spark session for data processing
        self.limit = limit   # Optional limit for data processing to restrict data size
        self.is_local = is_local  # Boolean flag to determine the data source (local or BigQuery)

        # Data files
        self.stackoverflow_posts_df = None
        self.tags_df = None
        self.posts_tag_wiki_excerpt_df = None
        self.posts_tag_wiki_df = None

        self.read_big_query_tables()  # Method call to read data tables
        self.clean_dfs()  # Method call to clean dataframes
        return

    # Read all the data sources
    def read_big_query_tables(self):
        if self.is_local:
            # Read from local CSV files
            self.stackoverflow_posts_df = self.spark_session.read.csv("datasets/stackoverflow_posts.csv", header = True, inferSchema = True)
            self.tags_df = self.spark_session.read.csv("datasets/tags.csv", header = True, inferSchema = True)
            self.posts_tag_wiki_excerpt_df = self.spark_session.read.csv("datasets/posts_tag_wiki_excerpt.csv", header = True, inferSchema = True)
            self.posts_tag_wiki_df = self.spark_session.read.csv("datasets/posts_tag_wiki.csv", header = True, inferSchema = True)
        else:
            # Read from BigQuery
            self.stackoverflow_posts_df = self.spark_session.read.format("bigquery").option("table", "bigquery-public-data.stackoverflow.stackoverflow_posts").load()
            self.tags_df = self.spark_session.read.format("bigquery").option("table", "bigquery-public-data.stackoverflow.tags").load()
            self.posts_tag_wiki_excerpt_df = self.spark_session.read.format("bigquery").option("table", "bigquery-public-data.stackoverflow.posts_tag_wiki_excerpt").load()
            self.posts_tag_wiki_df = self.spark_session.read.format("bigquery").option("table", "bigquery-public-data.stackoverflow.posts_tag_wiki").load()
        
        # Optionally limit the number of records processed
        if self.limit != None:
            self.stackoverflow_posts_df = self.stackoverflow_posts_df.select("id", "title", "body", "tags", "parent_id", "score").limit(self.limit)
            self.tags_df = self.tags_df.limit(self.limit)
            self.posts_tag_wiki_excerpt_df = self.posts_tag_wiki_excerpt_df.select("id", "body").limit(self.limit)
            self.posts_tag_wiki_df = self.posts_tag_wiki_df.select("id", "body").limit(self.limit)
        else:
            self.stackoverflow_posts_df = self.stackoverflow_posts_df.select("id", "title", "body", "tags", "parent_id", "score")
            self.posts_tag_wiki_excerpt_df = self.posts_tag_wiki_excerpt_df.select("id", "body")
            self.posts_tag_wiki_df = self.posts_tag_wiki_df.select("id", "body")
        
        return
    
    def clean_dfs(self):
        # Filter out posts that do not have any tags
        self.stackoverflow_posts_df = self.stackoverflow_posts_df.filter(self.stackoverflow_posts_df.tags != '')
        
        return

    # Returns a dataframe that serves as a lookup table for tag_id -> post_id
    def build_tag_post_df(self):
        tags_posts_df = self.stackoverflow_posts_df.select(self.stackoverflow_posts_df.id.alias("post_id"), explode(split(self.stackoverflow_posts_df.tags, "\\|")).alias("tag_name")).drop("tags")
        tags_posts_df = tags_posts_df.groupBy("tag_name").agg(collect_list("post_id").alias("post_ids"))
        tags_posts_df = tags_posts_df.withColumn("post_ids", concat_ws(",", tags_posts_df.post_ids))
        
        tags_df = self.tags_df.select(self.tags_df.id.alias("tag_id"), "tag_name", "count", "excerpt_post_id", "wiki_post_id")
        tag_post_df = tags_df.join(tags_posts_df, on = 'tag_name', how = 'inner')
        
        return tag_post_df
        
    # Returns a dataframe that serves as a lookup table for post_id -> post_title & post_body
    def build_post_text_df(self):
        posts_texts_df = self.stackoverflow_posts_df.select(self.stackoverflow_posts_df.id.alias("post_id"), "body", "title", "score")
        
        excerpts_texts_df = self.posts_tag_wiki_excerpt_df.select(self.posts_tag_wiki_excerpt_df.id.alias("post_id"), "body")
        excerpts_texts_df = excerpts_texts_df.withColumn("title", lit(None))
        excerpts_texts_df = excerpts_texts_df.withColumn("score", lit(None))

        wikis_texts_df = self.posts_tag_wiki_df.select(self.posts_tag_wiki_df.id.alias("post_id"), "body")
        wikis_texts_df = wikis_texts_df.withColumn("title", lit(None))
        wikis_texts_df = wikis_texts_df.withColumn("score", lit(None))
        
        post_text_df = posts_texts_df.union(excerpts_texts_df).union(wikis_texts_df)

        return post_text_df