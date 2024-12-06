from google.cloud import storage

class DataProc():
    def __init__(self):
        # Initialize a client to interact with Google Cloud Storage
        self.client = storage.Client()

    # List all buckets in the project
    def get_buckets(self,verbose=True):
        # Retrieves a list of all buckets in the Google Cloud project
        buckets = self.client.list_buckets()
        if verbose:
            # If verbose is True, print the names of all buckets
            for bucket in buckets: print(bucket.name)
        return buckets  # Return the list of bucket objects
    
    def download_files_from_bucket(self,bucket_name = "532-dataset",files = ["vectors.npz","tag_id_name.json"],out_path = "532/datasets"):
        # Fetch the specified bucket using its name
        bucket = self.client.get_bucket(bucket_name)
        for file in files:
            # Access the file (blob) within the specified bucket
            blob = bucket.blob(file)
            # Download the file to a local directory, constructing the path using the specified out_path
            blob.download_to_filename(out_path + '/' + file)

