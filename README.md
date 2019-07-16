# Predicting Chicago Taxi Fare

## Purpose of the Model
This model uses the Chicago Taxi Trip dataset. (https://www.kaggle.com/chicago/chicago-taxi-trips-bq)  

The goal of this project is to predict the fare price at the beginning of the ride, based on the available information: date and time of the ride, the pickup and drop off locations, the taxi driver’s anonymous ID and the company that provides the ride. 

Predicting the fare serves two purposes: First, it gives the passengers an expectation of the fare. Second, it prevents the meters’ mistakes in calculating the fare. In the dataset, we have seen impossible data, which could cause mistakes in fare calculations. For example,one trip was 15 minutes in duration but the distance traveled was 2000+ miles. The predicted fares could be used as a benchmark to evaluate the meters’ fares calculated to prevent mistakes. 

## Data Exploration and Data Analysis

The .ipynb file for data analysis is located in the exploratory-analysis folder:

/exploratory-analysis/chicago-taxi-fare-exploratory-analysis.ipynb 

<img src="/images/github_data_exploration.PNG" width="600">

### Data Sample
1M rows of data were extracted from Big Query using SQL. Apparently, the more data the better. However, since the purpose of this project is to evaluate the methodology, we only used a subset of the data in order to cut down costs.  

### Data Extraction and Basic Data Transformation Using Big Query
Thanks to Big Query’s powerful functions, some new features were created during data extraction for further explorations. These features were selected based on general experience in Machine Learning.

- Transforming the pickup and drop off data time from UTC to Chicago local
- Extracting the date, hour, day of the week, month and year from the pickup datetime
- Calculating the duration of the trip using beginning and ending time
- Marking trips that started or ended in an airport based on the community areas

~~~
# Pull data and feature engineering 
# The training data set will include 1M rows
# Feature engineering to generate new columns: 
#Year, Date, Hour, Weekday of the trip, whether the trip starts or end in an airport, duration of the trip in minutes.

sql="""
with tmp_tb as (
SELECT
    unique_key, 
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
    fare fare_dollars
FROM
    `bigquery-public-data.chicago_taxi_trips.taxi_trips` 
WHERE 
    fare > 0 and fare is not null and trip_miles > 0 and trip_miles is not null
ORDER BY 
    RAND()
LIMIT 1000000)
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

df = client.query(sql).to_dataframe()
~~~

### General Exploration of the Data
This step helps us understand:

- data types of each column

~~~
df.dtypes
~~~
<img src="/images/github_dtype.PNG" width="400">

- the outliers or implausible data, such as trips with duration of 15 minutes but distance of 1000+ miles which suggested that additional data filtering was needed

~~~
df.describe()
~~~
<img src="/images/github_describe.PNG" width="400">

- number of missing values for each column and across the dataset.

~~~
# missing values
print(df.isnull().sum())
~~~
<img src="/images/github_missing_values.PNG" width="400">

- the statistical properties of the columns, such as its distribution. 

~~~
# plot the distribution of fare
sns.kdeplot(df_4.fare_dollars, shade=True)
~~~

<img src="/images/github_fare_distribution.png" width="400">

Most of the fare were under $2000 per trip. There are a few big numbers, which needs further explorations on if they are legit fare prices or bad data.


~~~
# make a copy of the original data
df_2 = df.copy()

## Check how many rides are share rides. Include only non-shared rides
# If two unique_key have the same taxi_id and trip_start_timestamp, it is considered a shared ride

aggregations = {
    'unique_key':{
        'passengers': 'count'
    },
    'fare_dollars':{
        'max_fare': 'max', 
        'min_fare': 'min'
    }   
}

df_share_rides = df_2[['unique_key', 'taxi_id', 'trip_start_timestamp', 'fare_dollars']].groupby(['taxi_id', 'trip_start_timestamp']).agg(aggregations)
df_share_rides.columns = df_share_rides.columns.get_level_values(1)
df_share_rides.head()

# Filter to include only the rides that are not share rides
# For modeling purposes, only include the ones that are not share rides, to get an accurate estimation of the fare

df_3 = df_2.merge(df_share_rides, left_on=['taxi_id', 'trip_start_timestamp'], right_on=['taxi_id', 'trip_start_timestamp'], how='left')
df_4 = df_3.loc[df_3.passengers == 1]
df_4.shape
~~~

<img src="/images/github_share_rides.PNG" width="400">

It shows that only about 1% are share rides.


### Explore the Relationships Among Metrics
This step aimed to explore the relationships between different columns, including the relationships between different columns and fare.

Some feature engineering were used, such as extracting the hour of the day and day of the week from the pickup_datetime column. 

Tools used included statistical testing, such as correlation analysis, and visualization, including scatterplots, time-lapse animation and geographical plots.

~~~
df_4['mmdd'] = df_4.trip_start_timestamp.map(lambda x: x.strftime('%m-%d'))
fig = plt.figure()
ax1 = df_4.pivot_table('unique_key', index='hour', columns='year', aggfunc=lambda x: len(x.unique())).plot(figsize=(8,5))
plt.title('Number of Rides by Pickup Hour')
~~~

<img src="/images/github_rides_hour.png" width="400">

~~~
fig = plt.figure()
ax1 = df_4.pivot_table('fare_dollars', index='hour', columns='year').plot(figsize=(8, 5))
plt.title('Average Fare($) per Ride by Pickup Time')
~~~

<img src="/images/github_fare_hour.png" width="400">

~~~
fig = plt.figure()
ax1 = df_4.pivot_table('trip_miles', index='hour', columns='year').plot(figsize=(8, 5))
plt.title('Average Fare($) per Ride by Pickup Time')
~~~

<img src="/images/github_miles_hour.png" width="400">

~~~
fig = plt.figure()
ax1 = df_4.pivot_table('unique_key', index='mmdd', columns='year', aggfunc=lambda x: len(x.unique())).plot(figsize=(16,8))
plt.title('Number of Rides by Day')
~~~

<img src="/images/github_rides_day.png" width="400">

Explores the locations of the rides
~~~
# This function plots the rides on a map of chicago
def plot_data_points(longitude, latitude, data_frame, focus_point) :
    cvs = ds.Canvas(plot_width=500, plot_height=400)
    export  = partial(export_image, export_path = "export", background="black")
    agg = cvs.points(data_frame, longitude, latitude, ds.count())
    img = transfer_functions.shade(agg, cmap= hot, how='eq_hist')
    image_xpt  =  transfer_functions.dynspread(img, threshold=0.5, max_px=4)
    return export(image_xpt,"map")
~~~
~~~
# pickup locations
plot_data_points('pickup_longitude', 'pickup_latitude', df_4, 'unique_key')
~~~

<img src="/images/github_rides_by_pickup.png" width="400">

~~~
# plot of dropoff location
plot_data_points('dropoff_longitude', 'dropoff_latitude', df_4, 'unique_key')
~~~

<img src="/images/github_rides_by_dropoff.png" width="400">

Explore relationships between trip duration, trip length and fares

~~~
plt.scatter(df_4.trip_miles, df_4.fare_dollars, s = 1)
plt.xlabel('Trip Miles')
plt.ylabel('Fare($)')
~~~

<img src="/images/github_fare_miles.png" width="400">

It seems that there are a few rides with a much longer trip miles and fare amount. The following plot will zoom in to the trip with 500 miles or less, and fare dollars below 300.
 
Zooming in.

~~~
plt.scatter(df_4.trip_miles.loc[(df_4.trip_miles <=500) & (df_4.fare_dollars <=300)], 
            df_4.fare_dollars.loc[(df_4.trip_miles <=500) & (df_4.fare_dollars <=300)], 
            s = 1)
plt.xlabel('Trip Miles')
plt.ylabel('Fare($)')
~~~ 
 
Check how many rides have are 500 miles or less, and cost $300 or less.
~~~
df_4.trip_miles.loc[(df_4.trip_miles <=500) & (df_4.fare_dollars <=300)].shape
~~~

The results show (998528,). 99.7% of the trips were less than 500 miles and less than 300 dollars.

    
Two conclusions came out of this analysis:
1. The following factors were relevant in predicting the taxi fare: hour of the day when the trip started, day of the week, distance of the trip, locations of the pickup and drop off, where the trip started from or ended in an airport area.
2. There seemed to be two lines of services: premium services and regular services, based on the rates. 

In predicting whether a ride is a premium service, there are two hypothesis:
(1) a certain company mostly provides one type of service;
(2) a certain taxi_id always provides a premium or normal service.
It is also possible that a driver provides premium services when affiliated with a certain company, but might provide normal services when affiliated with another company.

The next steps is to explore which companies tend to provide premium services, and which drivers have provied premium services.


### Identifying Premium vs Regular Taxi Services
#### Clustering the Drivers
The analysis above suggested that there were two rates: a premium one and a regular one. Correctly identifying which group the driver falls into would increase the accuracy of fare prediction. A clustering analysis of the drivers was performed to reach this goal.

The drivers were profiled based on the metrics of their previous rides, such as the number of rides, total revenue, average dollars per mile, average miles per trip, etc. Analysis of this data showed that the drivers could be clustered based on the dollars per mile and average miles per trip, as drivers that provided premium services had higher average dollars per mile, and smaller miles per trip.

Then a KMean clustering analysis was performed. Different number of clusters (k) were tested and the results revealed that having two clusters provides a good balance between the complexity and explanatory power.

The result of the clustering was merged with the original data. 

#### Identifying Companies that Provide Premium Services
Analysis of the dollars per mile for each company showed a clear cut off among companies. Companies that had higher dollars per mile were marked in the column ‘is_luxury”.

##  Summary of Relevant Features
The features below were generated:
When extracting data from Big Query:
- Transforming the pickup and drop off data time from UTC to Chicago local
- Extracting the date, hour of the day, day of the week, month and year from the pickup datetime
- Calculating the duration of the trip using beginning and ending time
- Marking trips that started or ended in an airport based on the community areas

In Jupyter Notebook:
- Marking the service level, premium vs regular, of each driver based on the clustering results
- Marking the service level of the company based on the dollars per mile of the trips provided in the past

# Modeling

## Input
- pickup_latitude
- pickup_longitude
- dropoff_latitude
- dropoff_longitude
- weekday (day of the week)
- is_luxury(if the company tend to provide premium services)
- k2(cluster of the driver based on the clustering results)

Data that had mileage per hour (mph) of 90 or above, or the trip miles is above 300, or the fare is above 4 standard deviations, were removed. 

## Estimator: DNNLinearCombinedRegressor
The estimator used is DNNLinearCombinedRegressor from the tensorflow.estimator. This built-in estimator fits the use of this model and make it easy to use.

## Tuning Hyperparametesr using Hypertune
We used Hypertune to tune the number of nodes in each layer. The goal is to minimize the Root Mean Squared Error (RMSE). The parameters are set in the config.yaml file.

The job for hypertuning is training taxi_fare_5. 

~~~~
!gcloud ml-engine jobs submit training taxi_fare_5\
        --job-dir=gs://taxi_fare_4/job_folder/taxi_fare_model_5\
        --runtime-version=1.10 \
        --region=us-central1 \
        --module-name=trainer.task \
        --package-path=trainer \
        --config=config.yaml
~~~~

The job shows the hyperparameters and the performance indicator: rmse.


<img src="/images/github_hypertune.PNG" width="600">


The logs are saved in a cloud bucket:


<img src="/images/github_hypertune_log.PNG" width="600">


To check the results to see which model is the best:
~~~~
from googleapiclient import discovery
from google.oauth2 import service_account
import pandas as pd
import json

profect_id_name = 'hackathon1-183523'
project_id = 'projects/{}'.format(profect_id_name)
job_name = 'taxi_fare_5'
job_id = '{}/jobs/{}'.format(project_id, job_name)

# Build the service and make a request
ml = discovery.build('ml', 'v1')
request = ml.projects().jobs().get(name=job_id).execute()

# The best model
request['trainingOutput']['trials'][0]
~~~~


<img src="/images/github_best_results.PNG" width="600">


## Deployment and Predictions
The model used is taxi_fare_forecast_3, v4.
The job name for the prediction is taxi_fare_pred_4

The job for prediction is taxi_fare_model_45. 
~~~~
!gcloud ml-engine jobs submit prediction taxi_fare_pred_4 \
    --model=taxi_fare_forecast_3 \
    --input-paths=gs://taxi_fare_4/data/csv/test.csv \
    --output-path=gs://taxi_fare_4/data/prediction_output/ \
    --region=us-central1 \
    --data-format=TEXT \
    --signature-name=predict \
    --version=v4
~~~~


<img src="/images/github_pred_job.PNG" width="600">


If we click the job, we will see:


<img src="/images/github_pred_job_details.PNG" width="600">


## Results

Download the results and transform it into a dataframe.

~~~~
! gsutil cp gs://taxi_fare_4/data/prediction_output/prediction.results-00000-of-00001 pred_results.json
   
results_ls = []
for line in open('pred_results.txt', 'r'):
    results_ls.append(json.loads(line))
    
results = []
for item in results_ls:
    for data_item in item['predictions']:
        results.append(data_item)
~~~~

Calculate the rmse of the predicted results and actual data.

~~~~
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(results, test.fare_dollars))
print('RMSE is: ', rmse)
~~~~


<img src="/images/github_rmse.PNG" width="400">



# Reference
1. Google Cloud Platform: https://github.com/GoogleCloudPlatform/professional-services/tree/master/examples/cloudml-energy-price-forecasting

2. Kaggle: https://www.kaggle.com/chicago/chicago-taxi-trips-bq
