from google.cloud import bigquery

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
from sklearn.externals import joblib
from scipy import stats
from functools import partial

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
        default='gs://taxi_fare_3/data/csv/train_testtest.csv' 
    )
    args_parser.add_argument(
        '--pred_path',
        help='Location to prediction data.',
        default='gs://taxi_fare_3/data/csv/eval_testtest.csv'
    )
    args_parser.add_argument(
        '--driver_cluster_path',
        help='Location to validation data.',
        default='gs://taxi_fare_3/data/csv/driver_cluster.csv'
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
    args_parser.add_argument(
        '--pred_bq',
        help='The BigQuery Table for the pred data',
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


def rides_by_driver(df):
    '''
    examine the average miles per trip and dollars per mile by driver
    '''
    aggregations = {
        'unique_key':{
            'rides': 'count'
        },
        'fare':{
            'total_fare': 'sum'    
        },
        'trip_miles':{
            'total_miles': 'sum'
        }
    }

    by_driver = df.groupby('taxi_id').agg(aggregations)
    by_driver.columns = by_driver.columns.get_level_values(1)
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
    fare_upper = fare_upper_limit(df)
    
    # the upper limit of acceptable fare: 4 sd 
    count = sum(df.fare > fare_upper)
    print ('The upper limit of fare (4sd) is: $', fare_upper, '. ', count, 'rides are above the upper limit')

    # The upper limit of distance is arbitrary, because most trips are concentrated on the lower end, making the sd small
    count = sum(df.trip_miles > distance_upper_limit)
    print (count, 'rides are 300 miles or longer.')
    
    # Filter data based on on fare and trip miles, and mph, since we can't expect a future ride to exceed mph. It's illegal!
    df['mph'] = df.trip_miles/df.trip_minutes * 60
    df_filtered = df.loc[(df.mph < mph_upper_limit) & (df.fare <= fare_upper) & (df.trip_miles <= distance_upper_limit)]
                
    return (df_filtered)


def save_gcs(df, gcs_path):
    '''
    save the created data set as a csv file in the designated google cloud storage bucket
    '''
    df.to_csv('aaa.csv', index=False)

    
def get_gcs(gcs_path):
    '''
    Reads a file from the GCS bucket
    '''
    df = pd.read_csv(gcs_path)


def main():
    
    parameters = initialize_params()

    print ('Getting data from BigQuery database')
    client = bigquery.Client()
    
    # get training data
    train_sql = generate_sql(parameters.train_bq)    
    df = get_raw(client, train_sql)
    
    # cluster the drivers
    by_driver = cluster_drivers(df, parameters.clusters)
    
    # add drivers' cluster info
    df2 = merge_driver_cluster(df, by_driver)
    df3 = is_luxury(df2)
    
    # filter data
    df4 = filter_data(df3, parameters.distance_upper_limit, parameters.mph_upper_limit)
    df5 = df4[['hour','weekday','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude', 'k2','is_luxury','fare_dollars']]
    
    save_gcs(df5, parameters.training_path)
    print ('Training file sent to Google Cloud Storage.')
    
    ###  preparing the predition file
    print ('Prepping the file for predition.')
    pred_sql = generate_sql(parameters.pred_bq)
    pred = get_raw(client, pred_sql)
    
    # Merge with the driver's clusters
    pred2 = pred.merge(by_driver, left_on = ['taxi_id'], right_on = ['taxi_id'], how = 'left')
    
    # If a driver didn't exist in the training data set and hence no cluster, assign it to the popular group (most likely the normal taxi service group)    
    popular_k2 = by_driver.mode()['k2'][0]
    
    pred2.k2.fillna(popular_k2, inplace = True)
    pred2.k2 = pred2.k2.astype(np.int64)
    
    pred3 = is_luxury(pred2)
    pred4 = filter_data(pred3, parameters.distance_upper_limit, parameters.mph_upper_limit)
    pred5 = pred4[['hour','weekday','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude', 'k2','is_luxury','fare_dollars']]
    save_gcs(pred5, parameters.pred_path)
    print ('Pred file sent to Google Cloud Storage.')

    
if __name__ == '__main__':
    main()
    