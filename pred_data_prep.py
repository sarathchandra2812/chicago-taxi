from google.cloud import bigquery
from google.cloud import storage

import os, io, math, itertools, dsclient
import pandas as pd
import numpy as np
import argparse, json, os
from datetime import datetime
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from scipy import stats
from functools import partial
from io import StringIO # if going with no saving csv file


def initialize_params():
    """
    Sets parameters
    """
    args_parser = argparse.ArgumentParser()
     
    args_parser.add_argument(
        '--projectid',
        help='Project ID',
        default='hackathon1-183523'
    )
    args_parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models.',
        required=False
    )
    args_parser.add_argument(
        '--bucket_name',
        help='Name of the Google Cloud Storage bucket',
        default='taxi_fare_pp1' 
    )
    args_parser.add_argument(
        '--drivers_file_name',
        help='Specifying the file name for storing the file with drivers clusters.',
        default='drivers.csv' 
    )
    args_parser.add_argument(
        '--pred_file_name',
        help='Specifying the file name for storing the file for prediction.',
        default='pred.csv' 
    )
    args_parser.add_argument(
        '--pred_bq',
        help='The BigQuery Table for the pred data',
        default= 'bigquery-public-data.chicago_taxi_trips.taxi_trips' 
    )
    args_parser.add_argument(
        '--mph_upper_limit',
        help='The upper limit of the mph for a trip',
        default=90
    )    
    args_parser.add_argument(
        '--distance_upper_limit',
        help='The upper limit of the distance for a trip',
        default=500
    )

    return args_parser.parse_args()


def generate_sql(bq_table):
    sql1 = """with tmp_tb as (SELECT unique_key, 
        taxi_id, 
        DATETIME(trip_start_timestamp, 'America/Chicago') trip_start_timestamp, 
        DATETIME(trip_end_timestamp, 'America/Chicago') trip_end_timestamp, 
        trip_miles, 
        pickup_census_tract, 
        dropoff_census_tract, 
        pickup_community_area, 
        dropoff_community_area, 
        payment_type, 
        company, 
        pickup_latitude, 
        pickup_longitude, 
        dropoff_latitude, 
        dropoff_longitude,
        fare,
        fare/100 fare_dollars
        FROM `"""
     
    sql2 = """` WHERE 
        fare > 0 and fare is not null and trip_miles > 0 and trip_miles is not null
        ORDER BY 
        RAND()
        LIMIT 10000)
        SELECT *, 
        CAST(trip_start_timestamp AS DATE) trip_start_dt,
        CAST(trip_end_timestamp AS DATE) trip_end_dt,
        DATETIME_DIFF(trip_end_timestamp, trip_start_timestamp, MINUTE) trip_minutes,
        EXTRACT(YEAR FROM trip_start_timestamp) year,
        EXTRACT(MONTH FROM trip_start_timestamp) month,
        EXTRACT(DAY FROM trip_start_timestamp) day,
        EXTRACT(HOUR FROM trip_start_timestamp) hour,
        FORMAT_DATE('%a', DATE(trip_start_timestamp)) weekday,
        CASE WHEN (pickup_community_area IN (56, 64, 76)) OR (dropoff_community_area IN (56, 64, 76)) THEN 1 else 0 END is_airport
        FROM tmp_tb
        """
    
    sql = sql1 + bq_table + sql2
    
    return (sql)


def get_raw(client, sql):
    """
    get Chicago taxi fare raw data from BigQuery public dataset
    """
    raw = client.query(sql).to_dataframe()
    return raw


def merge_driver_cluster(df, by_driver_df):
    '''
    Merge the original ride info with the driver's clustesr
    '''
    merge_df = df.merge(by_driver_df, left_on=['taxi_id'], right_on='taxi_id', how='left')
    return (merge_df)


def fare_upper_limit(df):
    fare_upper = df.describe().fare['mean'] + 4 * df.describe().fare['std']
    return (fare_upper)


def is_luxury(df):
    df['is_luxury'] = np.where(df.company.isin(['Blue Ribbon Taxi Association Inc.', 'Suburban Dispatch LLC']), 1, 0)
    return (df)

    
def filter_data(df, distance_upper_limit, mph_upper_limit):
    '''
    Filter unrealistic data
    '''
    fare_upper = fare_upper_limit(df)
    
    # the upper limit of acceptable fare: 4 sd 
    count = sum(df.fare > fare_upper)

    # The upper limit of distance is arbitrary, because most trips are concentrated on the lower end, making the sd small
    count = sum(df.trip_miles > distance_upper_limit)
    
    # Filter data based on on fare and trip miles, and mph, since we can't expect a future ride to exceed mph. It's illegal!
    df['mph'] = df.trip_miles/df.trip_minutes * 60
    df_filtered = df.loc[(df.mph < mph_upper_limit) & (df.fare <= fare_upper) & (df.trip_miles <= distance_upper_limit)]
                
    return (df_filtered)


def check_bucket(storage_client, bucket_name):
    """Creates a new bucket."""
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except: 
        print ('The bucket ' + bucket_name + ' did not exist yet. Please enter the correct one. Thanks.')

def save_gcs(df, ds_client, gcs_path):
    '''
    save the created data set as a csv file in the designated google cloud storage bucket
    '''
    ds_client.write_csv(df, gcs_path)
    
    
def get_gcs(ds_client, gcs_path):
    '''
    Reads a file from the GCS bucket
    '''
    df = ds_client.read_csv(gcs_path)
    return (df)

    
def main():
    
    parameters = initialize_params()

    print ('Getting data from BigQuery database')
    client = bigquery.Client()
    client_ds = dsclient.Client(parameters.projectid)
    storage_client = storage.Client()
    
    check_bucket(storage_client, parameters.bucket_name)

    print ('Prepping the file for evaluation.')
    pred_sql = generate_sql(parameters.pred_bq)
    pred1 = get_raw(client, pred_sql)
    
    # get driver's clusters and merge with the test file
    drivers_gcs_path = 'gs://'+ parameters.bucket_name + '/data/csv/' + parameters.drivers_file_name
    by_driver = get_gcs(client_ds, drivers_gcs_path)
    pred2 = merge_driver_cluster(pred1, by_driver)
   
    # If a driver didn't exist in the training data set and hence no cluster, assign it to the popular group (most likely the normal taxi service group)    
    popular_k2 = by_driver.mode()['k2'][0]
    
    pred2.k2.fillna(popular_k2, inplace = True)
    pred2.k2 = pred2.k2.astype(np.int64)
    
    pred3 = is_luxury(pred2)
    pred4 = filter_data(pred3, parameters.distance_upper_limit, parameters.mph_upper_limit)
    pred5 = pred4[['hour','weekday','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude', 'k2','is_luxury']]
    
    new_header = pred5.iloc[0] #grab the first row for the header
    pred6 = pred5[1:] #take the data less the header row
    pred6.columns = new_header 

    pred_gcs_path = 'gs://'+ parameters.bucket_name + '/data/csv/' + parameters.pred_file_name
    save_gcs(pred6, client_ds, pred_gcs_path)
    print ('Pred file sent to Google Cloud Storage.' + pred_gcs_path)

    
if __name__ == '__main__':
    main()