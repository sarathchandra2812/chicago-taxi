from google.cloud import bigquery

import os, io, math, itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import hot
import seaborn as sns
from datetime import datetime
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import tensorflow as tf
from scipy import stats
import xgboost as xgb
from functools import partial
from bokeh.models import BoxZoomTool
from bokeh.plotting import figure, output_notebook, show
output_notebook()
import datashader as ds
from datashader.utils import export_image
from datashader import transfer_functions as transfer_functions

def get_raw(client, sql):
    """
    get Chicago taxi fare raw data from BigQuery public dataset
    """
    raw = client.query(sql).to_dataframe()
    return raw


def data_types(df):
    '''
    Show data types of each column
    '''
    print (df.dtypes)


def data_describe(df):
    '''
    Show the descriptive statistics (e.g., mean, min, max, count) of each column
    '''
    print (df.describe())    


def missing_values(df):
    '''
    Show the number of missing values for each column
    '''
    print(df.isnull().sum())

    
def share_rides(df):
    '''
    Check how many rides are share rides
    If two unique_key have the same taxi_id and trip_start_timestamp, it is considered a shared ride

    '''
    aggregations = {
        'unique_key':{
            'passengers': 'count'
        },
        'fare':{
            'max_fare': 'max', 
            'min_fare': 'min'
        }
    }
    
    df_2=df
    df_share_rides = df_2[['unique_key', 'taxi_id', 'trip_start_timestamp', 'fare']].groupby(['taxi_id', 'trip_start_timestamp']).agg(aggregations)
    df_share_rides.columns = df_share_rides.columns.get_level_values(1)
    
    # For modeling purposes, only include the ones that are share rides, to get an accurate estimation of the fare
    df_3 = df_2.merge(df_share_rides, left_on=['taxi_id', 'trip_start_timestamp'], right_on=['taxi_id', 'trip_start_timestamp'], how='left')
    df_4 = df_3.loc[df_3.passengers > 1]
    return (df_4)


def fare_distribution(df):
    '''
    plot the distribution of fare
    ''' 
    sns.kdeplot(df.fare_dollars, shade=True)

    
def rides_by_hour_plot(df):
    '''
    plot the number of rides for each hours of a day, for each year
    '''
    fig = plt.figure()
    ax1 = df.pivot_table('unique_key', index='hour', columns='year', aggfunc=lambda x: len(x.unique())).plot(figsize=(8,5))
    plt.title('Number of Rides by Pickup Hour')
    
    
def miles_by_hour_plot(df):
    '''
    plot the total miles of taxi rides for each hours of a day, for each year
    '''
    fig = plt.figure()
    ax1 = df.pivot_table('trip_miles', index='hour', columns='year').plot(figsize=(8, 5))
    plt.title('Average Fare($) per Ride by Pickup Time')    

    
def fare_by_hour_plot(df):
    '''
    plot the total miles of taxi rides for each hours of a day, for each year
    '''
    fig = plt.figure()
    ax1 = df.pivot_table('fare_dollars', index='hour', columns='year').plot(figsize=(8, 5))
    plt.title('Average Fare($) per Ride by Pickup Time')
    
def rides_by_date_plot(df):
    '''
    plot the rides for each day, for each year
    '''
    df['mmdd'] = df.trip_start_timestamp.map(lambda x: x.strftime('%m-%d'))
    
    fig = plt.figure()
    ax1 = df.pivot_table('unique_key', index='mmdd', columns='year', aggfunc=lambda x: len(x.unique())).plot(figsize=(16,8))
    plt.title('Number of Rides by Day')


def plot_data_points(longitude, latitude, data_frame, focus_point):
    '''
    This function plots the rides on a map of chicago
    '''
    cvs = ds.Canvas(plot_width=500, plot_height=400)
    export  = partial(export_image, export_path = "export", background="black")
    agg = cvs.points(data_frame, longitude, latitude, ds.count())
    img = transfer_functions.shade(agg, cmap= hot, how='eq_hist')
    image_xpt  =  transfer_functions.dynspread(img, threshold=0.5, max_px=4)
    
    return export(image_xpt, "map")


def plot_pickup(df):
    '''
    plot the number of rides by the starting points on a map
    '''
    return (plot_data_points('pickup_longitude', 'pickup_latitude', df, 'unique_key'))
    

def plot_dropoff(df):
    '''
    pot the number of rides by the dropoff points on a map
    '''
    return (plot_data_points('dropoff_longitude', 'dropoff_latitude', df, 'unique_key'))
   
        
def scatter_plot(dataframe, colX, colY):
    '''
    make scatter plots 
    '''
    plt.figure(figsize=(5, 5))
    sns.scatterplot(dataframe[colX], dataframe[colY])
    plt.show()
    
    
def rides_by_company(df):
    '''
    examine the average miles per trip and dollars per mile by each taxi company
    Output sorted by the descending order of the dollars per mile by company
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
    
    by_company = df.groupby('company').agg(aggregations)
    by_company.columns = by_company.columns.get_level_values(1)
    by_company['miles_per_trip'] = by_company.total_miles/by_company.rides
    by_company['dollars_per_mile'] = by_company.total_fare/by_company.total_miles
    by_company.sort_values(['dollars_per_mile'], ascending = False).head()
    
    return (by_company)


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
    
    
def plot_cluster_results(cluster_df):
    '''
    Plot the clustering results
    '''
    plt.scatter(cluster_df.dollars_per_mile, cluster_df.miles_per_trip, c = -cluster_df.k2)
    plt.xlabel('dollars per mile')
    plt.ylabel('miles per trip')
    plt.title('KMeans Results')