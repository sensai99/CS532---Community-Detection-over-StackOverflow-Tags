from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType

import os
# This environment variable is only needed for macOS, not required for Dataproc clusters
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import re
import string
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Define a custom path for nltk_data
nltk_data_dir = os.path.expanduser("/Users/Dell/nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Set NLTK data path
nltk.data.path.append(nltk_data_dir)

nltk.download('wordnet', download_dir = nltk_data_dir, quiet = True)
nltk.download('stopwords', download_dir = nltk_data_dir, quiet = True)
nltk.download('punkt', download_dir = nltk_data_dir, quiet = True)
nltk.download('punkt_tab', download_dir = nltk_data_dir, quiet=True)

stop_words = set(stopwords.words('english'))

# Define the UDF function outside the class
def clean_text(text):
    text = TextPreprocessor.remove_html_tags(text)
    text = TextPreprocessor.tokenize_text(text)
    text = TextPreprocessor.normalize_text(text)
    text = TextPreprocessor.remove_urls(text)
    text = TextPreprocessor.remove_stopwords(text)
    text = TextPreprocessor.stem_text(text)
    text = TextPreprocessor.lemmatize_text(text)
    return text

# Create UDF once
clean_text_udf = udf(clean_text, StringType())

class TextPreprocessor:
    def __init__(self, text_df):
        self.text_df = text_df
        return

    # Remove HTML tags
    @staticmethod
    def remove_html_tags(text):
        filtered_html_text = ""
        try:
            filtered_html_text = BeautifulSoup(text, "html.parser").get_text()
        except:
            filtered_html_text = ""
        return filtered_html_text
    # def remove_html_tags(text):
    #     return BeautifulSoup(text, "html.parser").get_text()

    @staticmethod
    def normalize_text(text):
        # Remove leading/trailing whitespaces whitespaces
        text = text.strip()
        
        # Lower the text
        text = text.lower() 
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7f]', ' ', text)
        
        return text
    
    # Remove URLs
    @staticmethod
    def remove_urls(text):
        return re.sub(r'http\S+', '', text)

    # Remove stopwords
    @staticmethod
    def remove_stopwords(text):
        return ' '.join([word for word in text.split() if word not in stop_words])

    # Tokenize text
    @staticmethod
    def tokenize_text(text):
        return ' '.join(word_tokenize(text))

    # Stemming
    @staticmethod
    def stem_text(text):
        stemmer = PorterStemmer()
        return ' '.join([stemmer.stem(word) for word in text.split()])

    # Lemmatization
    @staticmethod
    def lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    def preprocess_text(self):
        # Preprocess title & body of the posts
        text_df_processed = self.text_df.withColumn("body", clean_text_udf(col('body')))
        text_df_processed = text_df_processed.withColumn("title", clean_text_udf(col('title')))
        return text_df_processed
