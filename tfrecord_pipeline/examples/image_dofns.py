import io

import apache_beam as beam
from PIL import Image
from google.cloud.storage import Client


class ReadImages(beam.DoFn):
    gcs_client = None
    bucket = None

    def __init__(self, bucket_name):
        super().__init__()
        self.bucket_name = bucket_name

    def setup(self):
        self.gcs_client = Client()
        self.bucket = self.gcs_client.bucket(self.bucket_name)

    def process(self, element, *args, **kwargs):
        blob = self.bucket.blob(element)
        image_bytes = io.BytesIO(blob.download_as_bytes())
        im = Image.open(image_bytes)
        return [im]

    def teardown(self):
        self.gcs_client.close()
