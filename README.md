# CS532---Community-Detection-over-StackOverflow-Tags

## Dataset references
- [Kaggle dataset link](https://www.kaggle.com/datasets/stackoverflow/stackoverflow/data?select=tags)
- [StackExchange data explorer](https://data.stackexchange.com/)

## Hosting/Deployment
- Project is also deployed on the [dataproc cluster](https://console.cloud.google.com/welcome/new?authuser=1&hl=en&project=sytems-for-ds-532)


## Code Files
This project consists of several Python scripts and utility modules:
- `CommunityDetection.py`: Contains methods and classes to perform community detection on the dataset.
- `Dataproc_helper_utils.py`: Provides utility functions for managing and interfacing with Google Dataproc services.
- `Dataset.py`: Handles data loading, preprocessing, and structuring for analysis.
- `master.py`: Main script that initializes the process and coordinates the workflow.
- `TextPreprocessor.py`: Implements text cleaning and preprocessing functionalities.
- `TFIDF.py`: Manages the generation and manipulation of TF-IDF vectors from text data.
- `utils.py`: Includes additional helper functions and utilities used across the project.

## Getting Started
To run this project on your local machine or on a Dataproc cluster, ensure you have Python and PySpark installed, along with necessary dependencies like NLTK and scikit-learn for text processing and TF-IDF computation. Alternatively, use 
environment.yml to create the conda env using command - ```conda env create -f environment.yml

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request for review.

## License
This project is licensed under the MIT License.

## Contact
For bugs, features, or questions about the project, please file an issue in the GitHub repository or contact the project maintainers directly.