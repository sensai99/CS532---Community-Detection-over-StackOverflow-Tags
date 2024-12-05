from google.cloud import storage

class DataProc():
    def __init__(self):
        self.client = storage.Client()

    # List all buckets in the project
    def get_buckets(self,verbose=True):
        buckets = self.client.list_buckets()
        if verbose:
            for bucket in buckets: print(bucket.name)
        return buckets
    
    def download_files_from_bucket(self,bucket_name = "532-dataset",files = ["vectors.npz","tag_id_name.json"],out_path = "532/datasets"):
        bucket = self.client.get_bucket(bucket_name)
        for file in files:
            blob = bucket.blob(file)
            blob.download_to_filename(out_path + '/' + file)

