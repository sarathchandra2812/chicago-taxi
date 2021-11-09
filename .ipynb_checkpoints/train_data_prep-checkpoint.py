from google.cloud import bigquery
from google.cloud import storage

import os, io, math, itertools
import pandas as pd
import numpy as np
import argparse, json, os
from datetime import datetime
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
import joblib
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
        default='sarath-5'
    )
    args_parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models.',
        required=False
    )
    args_parser.add_argument(
        '--bucket_name',
        help='Name of the Google Cloud Storage bucket',
        default='sarath-5-chicago-taxi' 
    )
    args_parser.add_argument(
        '--training_file_name',
        help='Specifying the file name for storing the training file.',
        default='train.csv' 
    )
    args_parser.add_argument(
        '--drivers_file_name',
        help='Specifying the file name for storing the file with drivers clusters.',
        default='drivers.csv' 
    )
    args_parser.add_argument(
        '--eval_file_name',
        help='Specifying the file name for storing the eval file.',
        default='eval.csv' 
    )
    args_parser.add_argument(
        '--clusters',
        help='Number of driver clusters.',
        default=2
    )
    args_parser.add_argument(
        '--distance_upper_limit',
        help='The upper limit of the distance for a trip',
        default=500
    )
    args_parser.add_argument(
        '--mph_upper_limit',
        help='The upper limit of the mph for a trip',
        default=90
    )
    args_parser.add_argument(
        '--train_bq',
        help='The BigQuery Table for the training data',
        default= 'bigquery-public-data.chicago_taxi_trips.taxi_trips' 
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
        fare > 0  and fare is not null and trip_miles > 0 and trip_miles is not null
        ORDER BY 
        RAND()
        LIMIT 150000)
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


def check_bucket(storage_client, bucket_name):
    """Creates a new bucket."""
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except: 
        print ('The bucket ' + bucket_name + ' did not exist yet. Creating it now.')
        storage_client.create_bucket(bucket_name)
    
    
def rides_by_driver(df):
    '''
    examine the average miles per trip and dollars per mile by driver
    '''

    by_driver = df.groupby('taxi_id').agg(rides=('unique_key', 'count'), total_fare=('fare', 'sum'), total_miles=('trip_miles', 'sum'))
    #by_driver.columns = by_driver.columns.get_level_values(1)
    by_driver['miles_per_trip'] = by_driver.total_miles/by_driver.rides
    by_driver['dollars_per_mile'] = by_driver.total_fare/by_driver.total_miles
    by_driver.sort_values(['dollars_per_mile'], ascending = False).head()
    
    return by_driver


def standardize_by_driver(df):
    '''
    clustering drivers based on the two metrics of their past rides: miels per ride and dollars per mile
    '''
    by_driver_standard = stats.zscore(df[['dollars_per_mile']])
    return (by_driver_standard)


def cluster_drivers(df, clusters):
    '''
    cluster the drivers based on the dollars per mile. 
    It outputs a data frame showing the cluster ID for each driver.
    '''
    by_driver_df = rides_by_driver(df)
    
    #standardize
    by_driver_standard = standardize_by_driver(by_driver_df)
    
    #cluster
    k2 = KMeans(n_clusters = clusters, random_state = 0).fit(by_driver_standard)
    by_driver_df['k2'] = k2.labels_
    
    by_driver_df.reset_index(level=0, inplace=True)

    return (by_driver_df)


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
    # filter based on the upper limit of the fare
    fare_upper = fare_upper_limit(df)
    count = sum(df.fare > fare_upper)

    # filter based on the distance of the trip    
    count = sum(df.trip_miles > distance_upper_limit)
    
    # filter based on the mph
    df['mph'] = df.trip_miles/df.trip_minutes * 60
    df_filtered = df.loc[(df.mph < mph_upper_limit) & (df.fare <= fare_upper) & (df.trip_miles <= distance_upper_limit)]
                
    return (df_filtered)


def index_to_col(df):
    
    df.reset_index(level=0, inplace=True)
    return (df)


def save_gcs(df, gcs_path):
    '''
    save the created data set as a csv file in the designated google cloud storage bucket
    '''
    df.to_csv(gcs_path, index=False)
    
    
def main():
    
    parameters = initialize_params()
       
    client = bigquery.Client()
    storage_client = storage.Client()
        
    # check if the bucket exists. If not, create a new one. 
    check_bucket(storage_client, parameters.bucket_name)
    
    print ('Prepping the training file.')
    print ('Getting data from BigQuery database')

    # get training data
    train_sql = generate_sql(parameters.train_bq)    
    df = get_raw(client, train_sql)
        
    # cluster the drivers, and save the files to a gcs bucket
    by_driver = cluster_drivers(df, parameters.clusters)
    by_driver_2 = index_to_col(by_driver)
    
    drivers_gcs_path = 'gs://'+ parameters.bucket_name + '/data/csv/' + parameters.drivers_file_name
    save_gcs(by_driver_2, drivers_gcs_path)
    print ('Drivers file sent to ' + drivers_gcs_path + '.')

    # add drivers' cluster info
    df2 = merge_driver_cluster(df, by_driver)
    df3 = is_luxury(df2)
    
    # filter data
    df4 = filter_data(df3, parameters.distance_upper_limit, parameters.mph_upper_limit)
    df5 = df4[['hour','weekday','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude', 'k2','is_luxury','fare_dollars']]
    
    # saving the training data to GCS
    train_gcs_path = 'gs://'+ parameters.bucket_name + '/data/csv/' + parameters.training_file_name
    save_gcs(df5, train_gcs_path)
    print ('Training file sent to ' + train_gcs_path + '.')

    
if __name__ == '__main__':
    main()