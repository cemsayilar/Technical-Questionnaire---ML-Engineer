# This script about case study on pi works recurietment process.

# Goals
# The committee asks you to forecast the hourly bus usages for next week for each municipality
# Hence you can aggregate the two measurements for an hour by taking the max value (sum would not be a nice idea
# for the obvious reasons) for each hour, and you should model this data with a time series model of your selection.

# Note
# 1- It would be a nice idea to implement a very simple baseline model first, and then try to improve the accuracy by
#    introducing more complex methods eventually. The bare minimum requirement of the task is one simple baseline
#    and one complex method.
# 2- The committee says that they will use the last two weeks (starting from 2017-08-05 to 2017-08-19) as assessment
#    (test) data, hence your code should report the error (in the criterion you chose for the task) for the last two weeks.
#    You may use true values for the prediction of the last week of test data, then combine the error of the
#    first and last week of the test separately.
# 3- Keep in mind that the dataset has missing data, hence a suitable missing data interpolation would be useful.

# Imports
import pandas as pd
from datetime import datetime as dt
import numpy as np
from helpers import check_df, grab_col_names, cat_summary, num_summary
pd.set_option('display.max_columns', None)
pd.set_option('display.max_seq_items', None)
df = pd.read_csv('/Users/buraksayilar/Desktop/Self_Dev/İş Caseleri/IP_Works_Case.csv')
##################################### Explanatory Analysis ###########################################
check_df(df)



##################################### Feature Selection & Engineering ###########################################
# Creaeting test and train sets
# Convert the timestamp column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Fill in missing values using interpolation method
data = df.groupby('municipality_id').apply(lambda group: group.interpolate(method='ffill'))

# Aggregate the two measurements for an hour by taking the max value
data = data.groupby([pd.Grouper(key='timestamp', freq='H'), 'municipality_id']).max().reset_index()

# Step 2: Split the data into training and testing sets
df_train = data[data['timestamp'] < '2017-08-05']
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
df_test = data[data['timestamp'] >= '2017-08-05']
df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])

# I should groupby data with municipality_id in order to take daily and hourly bases
# usage of each municipality individually.

# Now I have max usage for every hour of the day in each municipality.
df_train.groupby(['municipality_id', 'date', 'hour'])['usage'].count()
# Maybe I can add change in the usage, max in a day, popular times or change patterns (if there are any).


##################################### Base Model with FB Prophet ###########################################

from fbprophet import Prophet

# Group the data by municipality and date, and take the mean usage value for each day.
df_train_agg = df_train.groupby(['municipality_id', 'date'])['max_usage'].mean().reset_index()

# Rename the columns to fit the Prophet library requirements.
df_train_agg = df_train_agg.rename(columns={'timestamp': 'ds', 'usage': 'y'}).drop(['municipality_id', 'total_capacity'], axis=1)

# Define the Prophet model and fit it to the data.
model = Prophet(yearly_seasonality=True, daily_seasonality=True)
model.fit(df_train_agg)

# Create a future dataframe for prediction.
future = model.make_future_dataframe(periods=168, freq='H', include_history=False)

# Make the prediction and get the forecasted values.
forecast = model.predict(future)
forecast = forecast[['ds', 'yhat']].rename(columns={'yhat': 'y'})
forecast.tail()
forecast.head(60)

df_train
df_test_true = df_test[(df_test['timestamp'] > '2017-08-05')]
forecast
#Calculate the error for the last two weeks of test data.

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(df_test_true, df_test_pred, squared=False)

print(f"RMSE for the last 2 weeks of test data with Prophet baseline model: {rmse:.2f}")






def predictor(dataframe, model_period = 408, municipality_id=int, model_prophet=False, model_complex=False):
    '''

    :param dataframe: main dataframe contains data from all municipality bus info
    :param model_period:
    :param municipality_id:
    :param model_prophet:
    :param model_complex:
    :return:
    '''
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
    dataframe = dataframe[dataframe['municipality_id'] == municipality_id]
    # Fill in missing values using interpolation method

    data = dataframe.groupby('municipality_id').apply(lambda group: group.interpolate(method='ffill'))

    # Aggregate the two measurements for an hour by taking the max value
    data = dataframe.groupby([pd.Grouper(key='timestamp', freq='H'), 'municipality_id']).max().reset_index()

    # Step 2: Split the data into training and testing sets
    df_train = data[data['timestamp'] < '2017-08-05']
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    df_test = data[data['timestamp'] >= '2017-08-05']
    df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])

    if model_prophet == True:
         from fbprophet import Prophet

         # Rename the columns to fit the Prophet library requirements.
         df_train_agg = df_train.rename(columns={'timestamp': 'ds', 'usage': 'y'}).drop(['municipality_id', 'total_capacity'], axis=1)

         # Define the Prophet model and fit it to the data.
         model = Prophet(yearly_seasonality=True, daily_seasonality=True)
         model.fit(df_train_agg)

         # Create a future dataframe for prediction.
         future = model.make_future_dataframe(periods=408, freq='H', include_history=False)

         # Make the prediction and get the forecasted values.
         forecast = model.predict(future)
         forecast = forecast[['ds', 'yhat']].rename(columns={'yhat': 'y'})
         forecast = forecast[forecast['ds'].dt.hour.isin([8, 9, 10, 11, 12, 13, 14, 15, 16])]
         forecast = forecast[forecast['ds'] >= '2017-08-05']
         df_test_true = df_test[(df_test['timestamp'] > '2017-08-05')].drop(['total_capacity', 'municipality_id'], axis=1)

         # Calculate the error for the last two weeks of test data.

         from sklearn.metrics import mean_squared_error
         df_test_true = df_test_true[df_test_true['timestamp'].isin(forecast['ds'])]
         rmse = mean_squared_error(df_test_true['usage'], forecast['y'], squared=False)

         return print(f"RMSE for the last 2 weeks of test data for municipality id: {municipality_id:.2f}, with Prophet baseline model: {rmse:.2f}")

predictor(df, model_prophet=True, municipality_id=3)
