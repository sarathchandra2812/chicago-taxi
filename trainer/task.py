#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This document follows Google's published example: 
# https://github.com/GoogleCloudPlatform/professional-services/tree/master/examples/cloudml-energy-price-forecasting
# ==============================================================================

"""
This file sets the parameters and runs the model.
"""

#import sys; sys.argv=['']; del sys

import argparse, json, os
import numpy as np
import pandas as pd
import tensorflow as tf
import inputs
import model
import datetime



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
    
    
    """
    Sets parameters
    """
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "-p", "--predict", 
        required=True, 
        help="Is it a prediction?")
    args_parser.add_argument(
        '--predict_file_name', 
        help='Path to the prediction file', 
        required=False
    )
    args_parser.add_argument(
        '--output_folder',
        help='GCS location to write checkpoints and export models.',
        default='gs://taxi_exps_4/'
    )
    args_parser.add_argument(
        '--job_dir',
        help='Where the latest model sits.',
        default='gs://taxi_exps_4/2020-05-24_04-16-25'
    )
    args_parser.add_argument(
        '--training_path',
        help='Location to training data.',
        default='gs://taxi_fare_pp1/data/csv/train.csv' 
    )
    args_parser.add_argument(
        '--validation_path',
        help='Location to validation data.',
        default='gs://taxi_fare_pp1/data/csv/eval.csv'
    )
    args_parser.add_argument(
        '--pred_path',
        help='Location to prediction data.',
        default='gs://taxi_fare_pp1/data/csv/pred.csv'
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

    parameters = args_parser.parse_args()   
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    folder_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    if parameters.predict=="True":
        print ('--------Predictions----------')
        data = pd.read_csv(parameters.pred_path)
        data = data.head()
        x_test = data[['hour', 'weekday', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'k2', 'is_luxury']]
        y = data['fare_dollars']
        
        run_config = tf.estimator.RunConfig()
        model_dir=parameters.job_dir
        run_config = run_config.replace(model_dir=model_dir)

        estimator = model.create_regressor(
            config = run_config, 
            parameters=parameters)
 
        
        pred_input_func = tf.estimator.inputs.pandas_input_fn(x = x_test, shuffle = False)

        predictions = estimator.predict(pred_input_func)
        
        for result in predictions:
            print (result)
    elif parameters.predict=="False":
        
        print ('--------Training---------')

        model_dir = os.path.join(parameters.output_folder, folder_timestamp, json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('trial', ''))
        
        run_config = tf.estimator.RunConfig(
            log_step_count_steps=1000,
            save_checkpoints_secs=120,
            keep_checkpoint_max=3,
            model_dir=model_dir
        )
        
        run_experiment(run_config, parameters)


if __name__ == '__main__':
    main()

