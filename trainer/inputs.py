
# This document follows Google's published example: 
# https://github.com/GoogleCloudPlatform/professional-services/tree/master/examples/cloudml-energy-price-forecasting
# ==============================================================================
"""
This file prepares the input data for modeling
"""

import multiprocessing
import pandas as pd
import numpy as np
import tensorflow as tf

TARGET_COLUMN = 'fare_dollars'
SHUFFLE_BUFFER_SIZE = 200

def parse_csv(record):
    """This function parses the .csv file
    Args:
        record: String representation of the record.
    Returns:
        A dictionary with all column names and values for the record.
    """
    column_names = ['hour', 
                    'weekday',
                    'pickup_latitude', 
                    'pickup_longitude',
                    'dropoff_latitude', 
                    'dropoff_longitude', 
                    'k2',
                    'is_luxury', 
                    'fare_dollars']
    
    header_def=[[0], [''], [0.0], [0.0], [0.0], [0.0], [0], [0], [0.0]]
    columns = tf.decode_csv(record, record_defaults=header_def)

    return dict(zip(column_names, columns))


def get_features_target_tuple(features):
    """
    This function returns the features and target
    """
    target = features.pop(TARGET_COLUMN, None)
    return features, target


def generate_input_fn(file_path, shuffle, batch_size, num_epochs):
    """
    This function preps data input.
    """
    def _input_fn():
        """
        This function returns a dictionary containing the features and the target.
        """
        num_threads = multiprocessing.cpu_count()
        dataset = tf.data.TextLineDataset(filenames=[file_path])
        dataset = dataset.skip(1)
        dataset = dataset.map(lambda x: parse_csv(
            tf.expand_dims(x, -1)), num_parallel_calls=num_threads)
        dataset = dataset.map(get_features_target_tuple,
                              num_parallel_calls=num_threads)
        if shuffle:
            dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        features, target = iterator.get_next()
        return features, target
    return _input_fn


def csv_serving_input_fn():
    """
    This function creates a ServingInputReceiver.
    """
    csv_row = tf.placeholder(
        dtype=tf.string
    )

    features = parse_csv(csv_row)
    features, _ = get_features_target_tuple(features)

    return tf.estimator.export.ServingInputReceiver(
        features=features,
        receiver_tensors={'csv_row': csv_row})


def get_train_spec(training_path, batch_size, max_steps):
    """
    This function creates a TrainSpec for the estimator
    """
    return tf.estimator.TrainSpec(
        input_fn=generate_input_fn(
            training_path,
            shuffle=True,
            batch_size=batch_size,
            num_epochs=None),
        max_steps=max_steps)

def get_eval_spec(validation_path, batch_size):
    """
    This function creates an EvalSpec for the estimaor.
    """
    exporter = tf.estimator.FinalExporter(
        'estimator',
        csv_serving_input_fn,
        as_text=False)

    return tf.estimator.EvalSpec(
        input_fn=generate_input_fn(
            validation_path,
            shuffle=False,
            batch_size=batch_size,
            num_epochs=None),
        exporters=[exporter],
        name='estimator-eval')