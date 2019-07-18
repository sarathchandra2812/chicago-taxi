#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This document follows Google's published example: 
# https://github.com/GoogleCloudPlatform/professional-services/tree/master/examples/cloudml-energy-price-forecasting
# ==============================================================================

"""
This file sets the parameters and runs the model.
"""

import sys; sys.argv=['']; del sys

import argparse, json, os
import numpy as np
import pandas as pd
import tensorflow as tf
import inputs
import model


def initialize_params():
    """
    Sets parameters
    """
    args_parser = argparse.ArgumentParser()
     
    args_parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models.',
        required=False
    )
    args_parser.add_argument(
        '--training_path',
        help='Location to training data.',
        default='gs://taxi_fare_3/data/csv/train.csv' 
    )
    args_parser.add_argument(
        '--validation_path',
        help='Location to validation data.',
        default='gs://taxi_fare_3/data/csv/eval.csv'
    )
    args_parser.add_argument(
        '--first_layer_size',
        help='First layer size.',
        default=256,
        type=int
    )
    args_parser.add_argument(
        '--number_layers',
        help='Number of hidden layers.',
        default=3,
        type=int
    )
    args_parser.add_argument(
        '--layer_reduction_fraction',
        help='Fraction to reduce layers in network.',
        default=0.5,
        type=float
    )
    args_parser.add_argument(
        '--learning_rate',
        help='Learning rate.',
        default=0.000001,
        type=float
    )
    args_parser.add_argument(
        '--batch_size',
        help='Training batch size.',
        default=64,
        type=int
    )
    args_parser.add_argument(
        '--eval_batch_size',
        help='Evaluation batch size.',
        default=168,
        type=int
    )
    args_parser.add_argument(
        '--max_steps',
        help='Maximum steps for training.',
        default=25000,
        type=int
    )
    args_parser.add_argument(
        '--nbuckets',
        help='Number of buckets for bucketing latitude and longitude.',
        default=20,
        type=int
    )
    return args_parser.parse_args()



def run_experiment(run_config, parameters):
    """
    This funtion runs the tensorflow model.
    """
    estimator = model.create_regressor(
        config = run_config, 
        parameters=parameters)
    train_spec = inputs.train_spec(
        parameters.training_path,
        parameters.batch_size,
        parameters.max_steps)
    eval_spec = inputs.eval_spec(
        parameters.validation_path,
        parameters.eval_batch_size)

    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )


def main():
    """
    This main function executes the job.
    """
    parameters = initialize_params()
    tf.logging.set_verbosity(tf.logging.INFO)
    model_dir = os.path.join('gs://taxi_exps_4/', json.loads(
        os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('trial', ''))
    
    run_config = tf.estimator.RunConfig(
        log_step_count_steps=1000,
        save_checkpoints_secs=120,
        keep_checkpoint_max=3,
        model_dir=model_dir
    )
    run_experiment(run_config, parameters)

if __name__ == '__main__':
    main()

