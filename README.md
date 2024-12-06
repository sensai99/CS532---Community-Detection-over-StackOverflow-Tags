# CS532---Community-Detection-over-StackOverflow-Tags

## About the Project 
In the vast ecosystem of StackOverflow, tags play a pivotal role in categorizing, organizing, and making content discoverable. With millions of questions and tags, finding and recommending the most relevant tags can be challenging yet critical for enhancing user experience and content accessibility.

This project, "Community Detection over StackOverflow Tags," aims to intelligently cluster related tags into communities using advanced data analytics and machine learning techniques on graph-based structures. By analyzing the relationships and similarities between tags based on their co-occurrence and textual descriptions, the system seeks to uncover inherent communities within the tags that represent deeper connections beyond mere keywords.

The insights generated through this community detection can significantly improve tag recommendations, streamline content navigation, and provide StackOverflow users with a more intuitive and efficient way to find the information they need. Moreover, these communities help in understanding the evolving trends and knowledge domains within the StackOverflow community, assisting in the dynamic organization of questions and answers.

Utilizing PySpark for data processing and Google Dataproc for scalable cloud deployment, this project leverages both to handle the large scale of data with efficiency. The resulting system not only enhances the user experience on StackOverflow but also contributes to the broader field of knowledge management in educational and professional tech communities.

## Environment Setup and Execution
Local Setup
1. Ensure Python and PySpark are installed on your local machine.
2. Install necessary Python libraries using:
```conda env create -f environment.yml```
3. Run master.py locally:
```python master.py ```


## Code Files Description
This project consists of several Python scripts and utility modules:
- `CommunityDetection.py`: Implements the core functionality for detecting communities using GraphFrames on Spark. It includes methods for loading data, creating similarity graphs, and performing community detection.
- `Dataproc_helper_utils.py`: Contains utility functions for interacting with Google Cloud Storage and managing data within Google Dataproc.
- `Dataset.py`:  Manages data loading and preprocessing, interfacing with both local data sources and BigQuery depending on the configuration.
- `master.py`: The entry point script that sets up the Spark session, initializes the process, and handles the execution of community detection.
- `TextPreprocessor.py`: Provides text cleaning and preprocessing functionalities to prepare text data for further analysis.
- `TFIDF.py`: Manages the creation and manipulation of TF-IDF vectors from text data to quantify tag similarity.
- `utils.py`: Includes helper functions such as cosine similarity calculations.


## Dataset references
- [Kaggle dataset link](https://www.kaggle.com/datasets/stackoverflow/stackoverflow/data?select=tags)
- [StackExchange data explorer](https://data.stackexchange.com/)


## Dataset Overview
This project utilizes a comprehensive dataset sourced from StackOverflow, a prominent platform for developers and IT professionals seeking to share knowledge and solve programming challenges. The dataset is critical for developing our community detection system, as it contains extensive information about tags used across millions of questions.

# Sources
1. StackOverflow Tags Dataset from Kaggle:

- Description: This dataset comprises a wide array of tags that users have applied to questions over the years. It includes details about each tag's usage frequency and associations with various technology stacks.
- Access: The dataset can be downloaded directly from Kaggle. Registration on Kaggle is required to access the data.
2. StackExchange Data Explorer:

- Description: For real-time and updated data, we utilize the StackExchange Data Explorer, which allows us to run queries against the current state of the StackOverflow database. This helps in analyzing trends and changes in tag usage over time.
- Access: Queries can be executed and customized via the StackExchange Data Explorer interface. This platform provides a flexible approach to data extraction based on specific needs and analysis goals.

# Data Structure
- Tags: Primary data element used for clustering. Each tag is associated with multiple questions, providing a rich source of contextual information.
- Tag Descriptions: Brief descriptions of what each tag represents, aiding in understanding the context and usage of the tag within the community.
- Usage Frequency: Each tag's occurrence frequency across the dataset, which helps in weighting tags in the clustering algorithms.

# Data Usage
The data is primarily used to:

- Construct a graph where tags represent nodes connected based on their co-occurrence in questions, weighted by their similarity derived from textual descriptions.
- Apply machine learning techniques to detect communities or clusters of closely related tags, enhancing the system's ability to recommend the most relevant tags to users.
- Analyze tag usage patterns to monitor the evolution of programming trends and community preferences over time.

# Data Processing
Data preprocessing is performed using ```TextPreprocessor.py```, which cleans and prepares the text data for further analysis, including:

- Removing irrelevant characters and stopwords.
- Normalizing text to lower case.
- Generating TF-IDF vectors to quantify tag similarity.

The cleaned and structured data is then processed using PySpark to leverage distributed computing for handling the dataset's volume and complexity efficiently.

## Data Preprocessing

Effective data preprocessing is crucial for ensuring the accuracy and efficiency of community detection in StackOverflow tags. This section describes the sequence of preprocessing steps applied to the dataset to optimize it for clustering analysis.

# Overview
The preprocessing pipeline transforms raw data from StackOverflow into a clean, structured format suitable for machine learning tasks. This involves text cleaning, normalization, and vectorization to facilitate sophisticated similarity measurements between tags.

# Steps
1. Data Cleaning:

- HTML Tag Removal: Any HTML tags embedded in the text descriptions are removed to ensure only textual content is analyzed.
- Punctuation and Special Characters: Non-alphanumeric characters, including punctuation, are stripped from the tag descriptions to reduce noise in the text data.

2. Text Normalization:

- Case Normalization: All text is converted to lowercase to maintain consistency and avoid duplications based on case differences.
- Tokenization: Text data is split into individual words or tokens, allowing for the analysis of the text at the word level.
- Stop Words Removal: Common words that do not contribute significant meaning to the tag descriptions, such as "the", "is", and "at", are removed.

3. Feature Extraction:

- TF-IDF Vectorization (TFIDF.py): The Term Frequency-Inverse Document Frequency (TF-IDF) technique is used to convert text data into a numeric form. TF-IDF highlights words that are more interesting, i.e., frequent in a particular document but rare across documents, which helps in distinguishing tags based on their unique descriptions.

4. Data Integration:

- Tag Co-occurrence Analysis: A matrix capturing the semantic closeness between tags is created based on the TF-IDF vectors. This matrix serves as the basis for defining relationships (edges) between tags (nodes) in the community detection algorithms.

## Graph Construction and Community Detection

# Overview
In this project, we build a tag correlation graph where each tag represents a node and the edge between tags is weighted with the cosine similarity of the constituent nodes. We utilize the Label Propagation Algorithm (LPA) to detect communities within the StackOverflow tags network. This enables us to perform quick recommendation for a particular tag and avoids search over entire tag space.

# Algorithm Description
The Label Propagation Algorithm operates based on the idea of spreading labels throughout the network and forming communities based on this label propagation process. Each node in the network starts with a unique label, and at every iteration, each node adopts the label that most of its neighbors currently have. This iterative process continues until convergence, typically when each node has the label that most of its neighbors have or when a maximum number of iterations is reached.

# Steps of the Label Propagation Algorithm
1. Initialization:

- Assign a unique label to each node in the graph (in this case, each StackOverflow tag is treated as a node).

2. Propagation:

- For each node, update its label to the one that the majority of its neighbors have (Here the popular label is decided based on the edge weights) . Ties can be broken uniformly at random.

3. Termination:

- Repeat the propagation step until no labels change or a predefined number of iterations is reached. This ensures that the algorithm terminates even if a perfect consensus isn't reached. For our experiments, we kept 20 iterations. 

# Advantages
- Scalability: LPA can handle large graphs efficiently because it requires only local information and has lower computational complexity.
- Simplicity: The algorithm does not require any prior information about the number or sizes of communities.
- Flexibility: It can be easily adapted or combined with other techniques to improve its effectiveness or to incorporate additional information.

# Implementation Details
The implementation of LPA in this project is done using PySpark to leverage distributed computing, allowing the algorithm to scale with the size of the dataset. We utilize the GraphFrames library in PySpark, which provides a scalable API for graph operations including the label propagation method. We ran our community detection experiments on Dataproc with multiple clusters to execute graph algorithms at scale with computation time.


# Challenges and Considerations
- Convergence: One of the challenges with LPA is ensuring that it converges, as it can oscillate in some cases. Implementing a maximum number of iterations helps mitigate this.
- Resolution: LPA may not always resolve fine-grained community structures, particularly in cases of overlapping communities or closely interconnected nodes.
- Post-Processing: After label propagation, some post-processing might be required to merge or split communities based on additional criteria or domain-specific knowledge.

# Conclusion
By employing the Label Propagation Algorithm, we aim to efficiently uncover the inherent community structure within StackOverflow tags, enhancing content discoverability and providing insights into the collaborative dynamics of the StackOverflow community.


## Tools and Libraries
The preprocessing steps are implemented using Python, with the following libraries:

- Pandas: For data manipulation and analysis.
- NLTK: For natural language processing tasks such as tokenization and stop words removal.
- Scikit-learn: For implementing TF-IDF vectorization.
- PySpark: Utilized for handling large datasets and performing operations in a distributed system environment, particularly useful when processing the full StackOverflow dataset.

## Execution
To execute the preprocessing steps, run the master.py script, which coordinates the workflow and calls other scripts/modules like TextPreprocessor.py and TFIDF.py to process the data accordingly. Ensure all dependencies are installed, and appropriate configurations are set up for PySpark to handle large-scale data effectively.


## Hosting/Deployment
- Project is also deployed on the [dataproc cluster](https://console.cloud.google.com/welcome/new?authuser=1&hl=en&project=sytems-for-ds-532)


## Model Testing


## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request for review.

## License
This project is licensed under the MIT License.

## Contact
For bugs, features, or questions about the project, please file an issue in the GitHub repository or contact the project maintainers directly.
