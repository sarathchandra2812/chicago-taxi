#!/usr/bin/env python
# coding: utf-8

# In[22]:

# This document follows Google's published example: 
# https://github.com/GoogleCloudPlatform/professional-services/tree/master/examples/cloudml-energy-price-forecasting
# ==============================================================================

"""
This function defines the Tensorflow model. 
"""

import math
import pickle
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import numpy as np
import pandas as pd

def create_regressor(config, parameters):
    """
    The model uses the built-in regressior DNNLinearCombinedRegressor.
    This function sets the features, evaluation and parameters for the DNNLinearCombinedRegressor.
    """
    
    INPUT_COLUMNS = [
        tf.feature_column.numeric_column('hour'),
        tf.feature_column.categorical_column_with_hash_bucket('weekday', hash_bucket_size = 7),
        tf.feature_column.numeric_column('pickup_latitude'),
        tf.feature_column.numeric_column('pickup_longitude'),
        tf.feature_column.numeric_column('dropoff_latitude'),
        tf.feature_column.numeric_column('dropoff_longitude'),
        tf.feature_column.categorical_column_with_identity('k2', num_buckets = 2),
        tf.feature_column.categorical_column_with_identity('is_luxury', num_buckets = 2)
    ]

    (hour, 
     dayofweek, 
     pickup_latitude, 
     pickup_longitude, 
     dropoff_latitude, 
     dropoff_longitude, 
     k2,
     is_luxury
    ) = INPUT_COLUMNS
    
    hourbuckets = np.linspace(0.0, 23.0, 24).tolist()
    buckets_hour = tf.feature_column.bucketized_column(hour, hourbuckets)    
    
    # bucket the longitude and latitude, based on the boundaries of Chicago
    latitude_buckets = np.linspace(41.6, 42.1, parameters.nbuckets).tolist()
    longitude_buckets = np.linspace(-88, -86,parameters.nbuckets).tolist()
    buckets_pickup_latitude = tf.feature_column.bucketized_column(pickup_latitude, latitude_buckets)
    buckets_dropoff_latitude = tf.feature_column.bucketized_column(dropoff_latitude, latitude_buckets)
    buckets_pickup_longitude = tf.feature_column.bucketized_column(pickup_longitude, longitude_buckets)
    buckets_dropoff_longitude = tf.feature_column.bucketized_column(dropoff_longitude, longitude_buckets)

    pickup = tf.feature_column.crossed_column([buckets_pickup_latitude, buckets_pickup_longitude], 
                                              parameters.nbuckets * parameters.nbuckets)
    dropoff = tf.feature_column.crossed_column([buckets_dropoff_latitude, buckets_dropoff_longitude], 
                                               parameters.nbuckets * parameters.nbuckets)
    loc_pair = tf.feature_column.crossed_column([pickup, dropoff], parameters.nbuckets ** 4 )
    
    wide_columns = [
        buckets_hour, dayofweek, k2, is_luxury
    ]

    deep_columns = [
        tf.feature_column.embedding_column(loc_pair, parameters.nbuckets),
        tf.feature_column.embedding_column(pickup, parameters.nbuckets),
        tf.feature_column.embedding_column(dropoff, parameters.nbuckets)
    ]
     
    
    layer = parameters.first_layer_size
    lfrac = parameters.layer_reduction_fraction
    nlayers = parameters.number_layers
    h_units = [layer]
    for _ in range(nlayers - 1):
        h_units.append(math.ceil(layer * lfrac))
        layer = h_units[-1]
        
        
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        linear_feature_columns = wide_columns,
        dnn_feature_columns = deep_columns,
        dnn_hidden_units = h_units,
        dnn_optimizer = tf.keras.optimizers.SGD(learning_rate=parameters.learning_rate),
        # dnn_optimizer=tf.train.ProximalGradientDescentOptimizer(learning_rate=parameters.learning_rate),
        # dnn_optimizer=tf.compat.v1.train.ProximalGradientDescentOptimizer(learning_rate=parameters.learning_rate),
        config=config
    )
    
    def root_mean_squared_error(labels, predictions):
        pred_values = predictions['predictions']
        # return {'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)}
        return {'rmse': tf.compat.v1.metrics.root_mean_squared_error(labels, pred_values)}
    

    estimator = tf.estimator.add_metrics(estimator, root_mean_squared_error)
    # estimator = tf.contrib.estimator.add_metrics(estimator, root_mean_squared_error)

    return estimator