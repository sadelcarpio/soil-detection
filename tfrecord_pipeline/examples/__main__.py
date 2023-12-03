import argparse
import logging
import os
import re

import apache_beam as beam
import apache_beam.ml.gcp.visionml as visionml
from apache_beam.io import fileio
from apache_beam.options.pipeline_options import PipelineOptions
from google.cloud import vision

from pipelines.examples.image_dofns import ReadImages

IMAGES_BUCKET_NAME = os.environ["IMAGES_BUCKET_NAME"]


def run_pipeline(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_date',
        default="11-11-2023",
        help="Date on which to process the images. Format DD-MM-YYYY"
    )
    known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)

    with beam.Pipeline(options=pipeline_options) as p:
        path_processing = (
                p
                | "Match .jpg filenames from GCS folder" >> fileio.MatchFiles(f"gs://{IMAGES_BUCKET_NAME}/"
                                                                              f"{known_args.batch_date}/*.jpg")
                | "Read said matches" >> fileio.ReadMatches()
                | beam.Reshuffle()
                | "Readable files" >> beam.Map(lambda x: x.metadata.path)
        )

        path_processing | "Write to storage" >> beam.Map(print)  # Print full gcs paths

        read_images = (
                path_processing
                | "Get only folder path (blob)" >> beam.Map(lambda x: re.split(rf"gs:\/\/{IMAGES_BUCKET_NAME}\/", x)[1])
                | "Read images from gcs path" >> beam.ParDo(ReadImages(bucket_name=IMAGES_BUCKET_NAME))
        )

        # Direct processing with visionml
        (
                path_processing
                | visionml.AnnotateImage(features=[{"type_": vision.Feature.Type.LABEL_DETECTION}])
                | beam.Map(print)
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)  # Shows INFO logs that are usually not shown
    run_pipeline()
