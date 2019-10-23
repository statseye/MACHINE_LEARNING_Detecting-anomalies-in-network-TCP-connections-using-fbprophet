### Data processing
### Author: Klaudia Łubian - StatsEye
### Date: 23/10/2019

# This codes it s continuoation of code "1. Combining and loading daily files.py". 

# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle

# Settings 
sns.set()
%matplotlib inline

pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Change working directory
import os
os.getcwd()
os.chdir(".../Time series/dane/all")
os.getcwd()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.plotting.register_matplotlib_converters()
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
import pickle
from fbprophet import Prophet
from fbprophet.plot import plot
import holidays
from sklearn.metrics import mean_absolute_error as mae


# Change working directory
os.getcwd()
os.chdir("C://Users/klubi/Desktop/learning/Python/Github/Time series/dane/all")
os.getcwd()

# Load pickled data

connects_min = pd.read_pickle("connects_min.pkl")

connects_min.info()
connects_min.head()

# check if it is possible to reduce the size of the file by changing the data type
connects_min['connects'].describe().apply(lambda x: format(x, 'f'))
# max = 69578. 
int_types = ["uint8", "uint16", "uint32"]
for it in int_types:
    print(np.iinfo(it))

# change to uint32
connects_min = pd.DataFrame(connects_min['connects'].astype('uint32'))
connects_min.info()
# memory usage reduced from 495.0 KB to 371.2 KB. 

# Assess the distribution of number of connections

connects_min['connects'].describe()

# In practice, histograms can be a substandard method for assessing the distribution of your data because they can be strongly affected by the number of bins that have been specified. Instead, kernel density plots represent a more effective way to view the distribution of your data. This is like the histogram, except a function is used to fit the distribution of observations and a nice, smooth line is used to summarize this distribution.

ax = connects_min.plot(kind = 'kde', figsize=(10,6))
ax.set_ylabel('Density plot of number of connections per minute', fontsize=10)
plt.show()

# Plot number of connections over time
# pd.plotting.register_matplotlib_converters() to avoid an erorr
connects_min.plot(figsize=(16,9))
plt.show();

# smooth by aggregating per hour
connects_min.resample('1H').sum().plot(figsize=(16,9));
plt.show();

print(connects_min.index[0])

# select 3 different  days to zoom in
connects_min[connects_min.index.day==2].plot(figsize=(16,9))
plt.title('Day: {}'.format(connects_min[connects_min.index.day==2].index[0].weekday_name))
plt.show();

connects_min[connects_min.index.day==4].plot(figsize=(16,9))
plt.title('Day: {}'.format(connects_min[connects_min.index.day==4].index[0].weekday_name))
plt.show();

connects_min[connects_min.index.day==7].plot(figsize=(16,9))
plt.title('Day: {}'.format(connects_min[connects_min.index.day==7].index[0].weekday_name))
plt.show();


# There seem to be:
# - increased activity during working hours (week days)
# - reduced activity during weekends
# - short peaks around 7pm (also during weekends)
# - short drops around 10pm (also during weekends)
# - slightly downward trend

### Asessing autocorrelation

# Display the autocorrelation plot of your time series. An autocorrelation of order 60 returns the correlation between a time series and its own values lagged by 60 time points (minutes).
# from statsmodels.graphics import tsaplots
plt.rcParams["figure.figsize"] = (12,9)
fig = tsaplots.plot_acf(connects_min['connects'], lags=60)
fig = tsaplots.plot_acf(connects_min['connects'], lags=120)

# Display the partial autocorrelation plot of your time series
fig = tsaplots.plot_pacf(connects_min['connects'], lags=60)
plt.show()
# Like autocorrelation, the partial autocorrelation function (PACF) measures the correlation coefficient between a time-series and lagged versions of itself. However, it extends upon this idea by also removing the effect of previous time points. 
#If partial autocorrelation values are close to 0, then values between observations and lagged observations are not correlated with one another. Inversely, partial autocorrelations with values close to 1 or -1 indicate that there exists strong positive or negative correlations between the lagged observations of the time series.


### Time series decomposition

'''
# First, let’s look at whether or not the data displays seasonality and a trend. To do this, we use the seasonal_decompose() function in the statsmodels.tsa.seasonal package. This function breaks down a time series into its core components: trend, seasonality, and random noise. You can rely on a method known as time-series decomposition to automatically extract and quantify the structure of time-series data. The statsmodels library provides an implementation of the naive, or classical, decomposition method in a function called seasonal_decompose(). It requires that you specify whether the model is additive or multiplicative. Both will produce a result and you must be careful to be critical when interpreting the result. A review of a plot of the time series and some summary statistics can often be a good start to get an idea of whether your time series problem looks additive or multiplicative.

# For this and for forecasting we will use Prophet
# https://towardsdatascience.com/a-quick-start-of-time-series-forecasting-with-a-practical-example-using-fb-prophet-31c4447a2274
# https://towardsdatascience.com/implementing-facebook-prophet-efficiently-c241305405a3

Prophet is framing the forecasting problem as a curve-fitting exercise rather than looking explicitly at the time based dependence of each observation within a time series.

# Before we get to the fun part, the goal of the blog I only want to show which hyper parameters to be on the lookout for as well as give some tips I have picked up. It is assumed that you have a relatively good understanding of Facebook Prophet.

# We are trying to model minutely data. As per October 2019, thereis not much on the Internet about the application of Prophet to minutely data. Found this comment from Facebook's employee: @iamshreeram we did some experiments with sub-daily (second) data, but paused those experiments awhile ago. If I see any significant results or findings I will report them here.
https://github.com/facebook/prophet/issues/226

# Assumptions: 
 - there is a minimum achieavable point due to contraints = 0, hence the 'floor' parameter should be set to 0. Domain expert says there is no saturating maximum , but to use a logistic growth trend with a saturating minimum, a maximum capacity must also be specified. We will check for the max historical value.  
 - It can be seen from the plot that there is roughly constant level (the mean of 19942 connections per minute), no changepoints are visible. The seasonal fluctuation and random fluctuations roughly are constant in size over time. This suggests that it’s probably appropriate to describe the data using an additive model which is Prophet built on.
 - holidays will have an effect on the timeseries (as can be concluded based on reduced activity during the weekends), use a built-in collection of country-specific holidays using the add_country_holidays method :
m = Prophet(holidays=holidays)
m.add_country_holidays(country_name='PL')
m.fit(df)
holidays_prior_scale: this parameter determines how much of an effect holidays should have on your predictions. So for instance when you are dealing with population predictions and you know holidays will have a big effect, try big values. Normally values between 20 and 40 will work, otherwise the default value of 10 usually works quite well. 
- seasonality_mode: default value here is additive with multiplicative being the other option. You will use additive when your seasonality trend should be “constant” over the entire period.
- seasonality_prior_scale parameter. This parameter will again allow your seasonalities to be more flexible. I have found values between 10 and 25 to work well here, depending on how much seasonality you notice in the components plot.

'''

# Step 0. Split data set into training set and test set. 
# Here we would like to use training data set to predict next 4 days. 

print("Number of minutes in training dataset:", 0.8*len(connects_min))
print("This translets to number of days:", 0.8*len(connects_min)/(60*24))
connects_min.info()
first_date = connects_min.index[1]
last_date = connects_min.index[-1]
last_date

# Decode on the end of the training dataset
train_last = '2019-09-19 23:59:00'
        
# set train and test datasets
train = connects_min[connects_min.index <= train_last]
test = connects_min[connects_min.index > train_last]

train.tail()
test.head()

# Visualize the split between train and test
plt.plot(train.index, train.connects, label = 'train')
plt.plot(test.index, test.connects, label = 'test')
plt.legend();


# Step 1. Start with weekly_seasonality and daily_seasonality, let the Prophet detect seasonality and treat it as a first, 'benchmark' model

fb_train = train.reset_index().copy()
fb_train.columns = ['ds', 'y']
fb_train.head()

fb_test = test.reset_index()
fb_test.columns = ['ds', 'y']
fb_test.info()
fb_test.head()

# check max historical value
print("Max historical value", connects_min.connects.max())

# set saturating min and max
fb_train['cap'] = 80000
fb_train['floor'] = 0

# Define first model and fit it using prophet
# Note that when sub-daily data are used, daily seasonality will automatically be fit.
m1 = Prophet(growth='logistic', 
             daily_seasonality = True, 
             weekly_seasonality = True, 
             yearly_seasonality = True)
m1.add_country_holidays(country_name= 'DK')
m1.fit(fb_train)

# Prepare prediction dataframe ( = time frame of test dataset)
len(fb_test)
num_points_to_pred = len(fb_test)

# freq='min' means 1 minute
future = m1.make_future_dataframe(periods = num_points_to_pred, freq='min', include_history=True)
future['cap'] = 80000
future['floor'] = 0
future.describe()

# Predict
forecast = m1.predict(future)




# Define a function for evaluating the results
# from sklearn.metrics import mean_absolute_error as mae
def plot_forecast(test, forecast, model_string, m):
    forecast = forecast[-num_points_to_pred:]
    score = np.round(mae(test.y, forecast.yhat),0)
    plt.rcParams['figure.figsize']=15,10
    plt.title('Forecast last {} points (minutes).\nmae={}\nProphet'.format(num_points_to_pred, score))
    plt.plot(test.ds, test.y, label='test: observed')
    plt.plot(test.ds, forecast.yhat, label='forecast')
    plt.legend();
    plt.savefig('C://Users/klubi/Desktop/learning/Python/Github/Time series/outputs/{}_mae={} .png'.format(model_string, score))
    
    m.plot(forecast)
    
    m.plot_components(forecast).savefig('C://Users/klubi/Desktop/learning/Python/Github/Time series/outputs/{}_components_mae={} .png'.format(model_string, score));

# Evaluate and plot
plot_forecast(fb_test, forecast, 'm1', m1)

# Shortcommings of the prediction:
# - increased activity during working hours (week days): daily seasonality needs to be made more flexible, use seasonality_prior_scale parameter,
# - reduced activity during weekends: model unnecessarily assumes increased activity during working hours: need to account for weekend seasonality
# - short large peaks around 7pm (also during weekends): model underestimates them: again, seasonality_prior_scale parameter, 
# the seasonality needs to fit higher-frequency changes, and generally be less smooth. The Fourier order can be specified for each built-in seasonality when instantiating the model, i.e. daily_seasonality = ... e.g. 40 (default=10)
# - short drops around 10pm (also during weekends) - underestimated

# Looking at the components plot:
# Weekly trend apparently detected - weekend drop can be observed. 
# Not sure a yearly seasonality is needed - we don't have enough data, just one month. 
# Daily trend is defeinitely needed, but, as mentioned above, needs to be more sensitive to big spikes/ drops. 


# Step 2. Set a daily seasonal pattern that is different on weekends vs. on weekdays. 

'''
Set a daily seasonal pattern that is different on weekends vs. on weekdays. These types of seasonalities can be modeled using conditional seasonalities. By doing this you get more power and control over seasonality.

period: Float number of days in one period. If the period is set to 30 then you tell the model that what happened at a certain point is likely to happen again in 30 days.

The other parameter you can tweak using this technique is the number of Fourier components (fourier_order) each seasonality is composed of. Increasing the number of Fourier components allows the seasonality to change more quickly (at risk of overfitting). It;s about frequency of bumps. Default value is 10. You can try values that range from 10 to 25.

There is also a seasonality_prior_scale parameter. This parameter will again allow your seasonalities to be more flexible. Prior scales are defined to tell the model how strongly it needs to consider the seasonal/holiday components while fitting and forecasting. Defaults to 10). I have found values between 10 and 25 to work well here, depending on how much seasonality you notice in the components plot.
'''

# First add a boolean column to the dataframe that indicates whether each date is during the weekend
def is_weekend(ds):
    date = pd.to_datetime(ds)
    return (date.weekday() == 5 or date.weekday() == 6)

fb_train['weekends'] = fb_train['ds'].apply(is_weekend)
fb_train['week_days'] = ~fb_train['ds'].apply(is_weekend)
fb_train.head()

fb_train[fb_train['weekends'] == True].head()
fb_train[fb_train['week_days'] == True].head()

fb_train['weekends'].describe()
fb_train['week_days'].describe()


# Then we disable the built-in daily seasonality, and replace it with two daily seasonalities that have these columns specified as a condition. This means that the seasonality will only be applied to dates where the condition_name column is True. 

m2 = Prophet(growth='logistic', 
             daily_seasonality = False, 
             weekly_seasonality = True, 
             yearly_seasonality = True)
m2.add_country_holidays(country_name= 'DK')
m2.add_seasonality(name='daily_weekdays', 
                   period=1, #number of days in one period 
                   fourier_order=100, # fit higher-frequency changes
                   prior_scale=100, # allow more flexibility, need to detect daily peaks and dumps better  
                   condition_name='week_days')
m2.add_seasonality(name='daily_weekends', 
                   period=1, #number of days in one period 
                   fourier_order=100, # fit higher-frequency changes
                   prior_scale=100, # allow more flexibility, need to detect daily peaks and dumps better
                   condition_name='weekends')
m2.fit(fb_train)

# We must also add the column to the future dataframe for which we are making predictions.

future2 = future.copy()
future2['weekends'] = future['ds'].apply(is_weekend)
future2['week_days'] = ~future['ds'].apply(is_weekend)
future2.head()
future2.tail()

# Predict
forecast2 = m2.predict(future2)

# Evaluate and plot
plot_forecast(fb_test, forecast2, 'm2', m2)

# zoom in
plt.rcParams['figure.figsize']=25,10
plt.title('Forecast last {} points (minutes)'.format(num_points_to_pred))
plt.plot(fb_test.ds, fb_test.y, label='test: observed')
plt.plot(fb_test.ds, forecast2[-num_points_to_pred:].yhat, label='forecast')
plt.legend();


# Add hourly seasonality

m3 = Prophet(growth='logistic', 
             daily_seasonality = False, 
             weekly_seasonality = True, 
             yearly_seasonality = True)
m3.add_country_holidays(country_name= 'DK')
m3.add_seasonality(name='daily_weekdays', 
                   period=1, #number of days in one period 
                   fourier_order=100, # fit higher-frequency changes
                   prior_scale=100, # allow more flexibility, need to detect daily peaks and dumps better  
                   condition_name='week_days')
m3.add_seasonality(name='daily_weekends', 
                   period=1, #number of days in one period 
                   fourier_order=100, # fit higher-frequency changes
                   prior_scale=100, # allow more flexibility, need to detect daily peaks and dumps better
                   condition_name='weekends')
m3.add_seasonality(name='hourly', 
                   period=1/24, #day/number of hours per day
                   fourier_order=30, # fit higher-frequency changes
                   prior_scale=70) # allow more flexibility
m3.fit(fb_train)
   
# Predict
forecast3 = m3.predict(future2)

# Evaluate and plot
plot_forecast(fb_test, forecast3, 'm3', m3)

m3.plot(forecast3)

# zoom in
plt.rcParams['figure.figsize']=25,10
plt.title('Forecast last {} points (minutes)'.format(num_points_to_pred))
plt.plot(fb_test.ds, fb_test.y, label='test: observed')
plt.plot(fb_test.ds, forecast3[-num_points_to_pred:].yhat, label='forecast')
plt.legend();

# Peaks are not covered even if we increase the pror_scale parameter. Maybe introducing an additional effect (regressor) will help?

# Zoom in to investigate the evening pick
# week day
connects_min['2019-09-04 18:00':'2019-09-04 20:00'].plot(figsize=(16,9))
plt.title('Week day: 2019-09-04 18:00 - 2019-09-04 20:00')
plt.show();
# between 19:00 and 19:40

# weekend
connects_min['2019-09-15 18:00':'2019-09-15 20:00'].plot(figsize=(16,9))
plt.title('Weekend: 2019-09-15 18:00 - 2019-09-15 20:00')
plt.show();
# between 19:00 and 19:40

# Additional regressors can be added to the linear part of the model using the add_regressor method or function. A column with the regressor value will need to be present in both the fitting and prediction dataframes. The add_regressor function has optional argument for specifying the prior scale, by default this parameter is 10, which provides very little regularization. Reducing this parameter dampens the regressor's effect. 

from datetime import datetime

def evening_pick(ds):
    date = pd.to_datetime(ds)
    if date.hour == 19:
        if date.minute >= 0 and date.minute <= 40:
            return 1
    else:
        return 0

fb_train['evening_pick'] = fb_train['ds'].apply(evening_pick)
fb_train['evening_pick'] = fb_train['evening_pick'].fillna(0)


m4 = Prophet(growth='logistic', 
             daily_seasonality = False, 
             weekly_seasonality = True, 
             yearly_seasonality = True)
m4.add_country_holidays(country_name= 'DK')
m4.add_seasonality(name='daily_weekdays', 
                   period=1, #number of days in one period 
                   fourier_order=100, # fit higher-frequency changes
                   prior_scale=100, # allow more flexibility, need to detect daily peaks and dumps better  
                   condition_name='week_days')
m4.add_seasonality(name='daily_weekends', 
                   period=1, #number of days in one period 
                   fourier_order=100, # fit higher-frequency changes
                   prior_scale=100, # allow more flexibility, need to detect daily peaks and dumps better
                   condition_name='weekends')
m4.add_seasonality(name='hourly', 
                   period=1/24, #day/number of hours per day
                   fourier_order=30, # fit higher-frequency changes
                   prior_scale=70) # allow more flexibility
m4.add_regressor('evening_pick') # default value proved to offer best results

m4.fit(fb_train)
   
# Predict
future3 = future2.copy()

future3['evening_pick'] = future3['ds'].apply(evening_pick)
future3['evening_pick'] = future3['evening_pick'].fillna(0)

forecast4 = m4.predict(future3)

# Evaluate and plot
plot_forecast(fb_test, forecast4, 'm4', m4)


# After checking results for different prior_scale levels set for the .add_regressor function, the defauly avlues offers the only solutions that reduces MAE with comparred to m3. However, we are still not able to predict the highest picks. In order to avoid detecting them as outliers, we will change the uncertainty interval to produce a confidence interval around the forecast from the deafult .80 to 0.99. 

m5 = Prophet(growth='logistic', 
             daily_seasonality = False, 
             weekly_seasonality = True, 
             yearly_seasonality = True, 
             interval_width=0.99)
m5.add_country_holidays(country_name= 'DK')
m5.add_seasonality(name='daily_weekdays', 
                   period=1, #number of days in one period 
                   fourier_order=100, # fit higher-frequency changes
                   prior_scale=100, # allow more flexibility, need to detect daily peaks and dumps better  
                   condition_name='week_days')
m5.add_seasonality(name='daily_weekends', 
                   period=1, #number of days in one period 
                   fourier_order=100, # fit higher-frequency changes
                   prior_scale=100, # allow more flexibility, need to detect daily peaks and dumps better
                   condition_name='weekends')
m5.add_seasonality(name='hourly', 
                   period=1/24, #day/number of hours per day
                   fourier_order=30, # fit higher-frequency changes
                   prior_scale=70) # allow more flexibility
m5.add_regressor('evening_pick') # default value proved to offer best results

m5.fit(fb_train)
   
# Predict
forecast5 = m5.predict(future3)

# Evaluate and plot
plot_forecast(fb_test, forecast5, 'm5', m5)


# preditions for last hour using interval_width=0.80. 
forecast4[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(60)
m4.plot(forecast4);

# preditions for last hour using interval_width=0.99. 
forecast5[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(60)
m5.plot(forecast5);

# Set yearly seasonality to False (just try)
m6 = Prophet(growth='logistic', 
             daily_seasonality = False, 
             weekly_seasonality = True, 
             yearly_seasonality = False, 
             interval_width=0.99)
m6.add_country_holidays(country_name= 'DK')
m6.add_seasonality(name='daily_weekdays', 
                   period=1, #number of days in one period 
                   fourier_order=100, # fit higher-frequency changes
                   prior_scale=100, # allow more flexibility, need to detect daily peaks and dumps better  
                   condition_name='week_days')
m6.add_seasonality(name='daily_weekends', 
                   period=1, #number of days in one period 
                   fourier_order=100, # fit higher-frequency changes
                   prior_scale=100, # allow more flexibility, need to detect daily peaks and dumps better
                   condition_name='weekends')
m6.add_seasonality(name='hourly', 
                   period=1/24, #day/number of hours per day
                   fourier_order=30, # fit higher-frequency changes
                   prior_scale=70) # allow more flexibility
m6.add_regressor('evening_pick') # default value proved to offer best results

m6.fit(fb_train)
   
# Predict
forecast6 = m6.predict(future3)

# Evaluate and plot
plot_forecast(fb_test, forecast6, 'm6', m6)

# better results, but should be worse if our timeframe includes more than one month


### Anomaly detection

# Set as outliers everything higher than the top and lower the bottom of the model boundary. In addition, the set the importance of outlier as based on how far the dot from the boundary.

# add observed values to forecast dataset (name the column 'fact')

connects_min.describe()
forecast5['fact'] = connects_min['connects'].reset_index(drop = True)
forecast5.info()

def detect_anomalies(forecast):
    forecasted = forecast[['ds','trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()

    forecasted['anomaly'] = 0
    forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
    forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1

    #anomaly importances (as % difference), valid only for anomalies
    forecasted.loc[forecasted['anomaly'] ==1, 'importance'] = \
        (forecasted['fact'] - forecasted['yhat_upper'])/forecast['fact']
    forecasted.loc[forecasted['anomaly'] ==-1, 'importance'] = \
        (forecasted['yhat_lower'] - forecasted['fact'])/forecast['fact']
        
    #absolute difference, valid only for anomalies
    forecasted.loc[forecasted['anomaly'] ==1, 'diff_upper'] = \
        (forecasted['fact'] - forecasted['yhat_upper']).abs()
    forecasted.loc[forecasted['anomaly'] ==-1, 'diff_lower'] = \
        (forecasted['fact'] - forecasted['yhat_lower']).abs()
    
    return forecasted

# Detect outliers by applying detect_anomalies() function
pred = detect_anomalies(forecast5)

pred.head()
pred.describe()

# Are there any anomalies detected in the test dataset?
pred[pred.ds > train_last].anomaly.value_counts()
print('365 out of 5640 values in test dataset were flagged as anomalies, which is {} %'.format(round(365/5640*100),1))
# Are there any anomalies detected in the train dataset?
pred[pred.ds <= train_last].anomaly.value_counts()
print('494 out of 31680 values in train dataset were flagged as anomalies, which is {} %'.format(round(494/31680*100),1))

# only test dataset
pred_test = pred[pred.ds > train_last]
pred_test.describe()
# most of the anomalies were 'outliers' above the upper bound of prediction (331), but the mean absolute difference between the observed values and bound is lower for the 'picks' (1745) than for the downs (2952), which confirms that our model is better fit to picks down to unexpected 'downs' - which is good, we do not think the downs would be problematic, we are after detecting unexpected picks in the number of connections. 

pred_test['diff_upper'].hist(bins=100);
pred_test['importance'].hist(bins=100);

plt.rcParams['figure.figsize']=20,8
plt.plot(pred_test.ds, pred_test.fact, label='test: observed')
plt.plot(pred_test.ds, pred_test.yhat_upper, label='yhat_upper')
plt.legend();

# filter anomalies > yhat_upper only
anomalies = pred[pred['anomaly'] == 1]
anomalies.describe()
# mean importance of these anomalies is 0.08, which means that their values exceed the upper bound by 8% on average. Max = 55%!

# Option 1
# Draw a border at 0.95 percentile (top 5% of the highest importance will be flagged)
anomalies['anomaly_check'] = 0
anomalies.loc[anomalies['importance'] >= np.percentile(anomalies['importance'], 95), 'anomaly_check'] = 1 

anomalies.describe()
anomalies['anomaly'].sum()
anomalies['anomaly_check'].sum()
# 33 anomalies flagged (all time series), foltered out from the original 659 anomalies flagged)

anomalies.loc[anomalies['anomaly_check'] ==1, 'diff_upper'].hist(bins=100);


'''
# Option 2
# an alternative - remove the occurances of anomalies that happened during evening picks, i.e. 19:00 - 19:40

anomalies['evening_pick'] = anomalies['ds'].apply(evening_pick)
anomalies['evening_pick'] = anomalies['evening_pick'].fillna(0)
anomalies.describe()
# 19% of anomalies occured during the pick time

# Exclude 'evening_pick'
anomalies['anomaly_check2'] = 0
anomalies.loc[anomalies['evening_pick'] == 0, 'anomaly_check2'] = 1
anomalies['anomaly_check2'].sum()
# number of anomalies reduced to 532 from original 659. 
anomalies.loc[anomalies['anomaly_check2'] ==1, 'diff_upper'].hist(bins=100);

# Now select .95 percentile
anomalies.loc[anomalies['importance'] < np.percentile(anomalies['importance'], 95), 'anomaly_check2'] = 0 
anomalies['anomaly_check2'].sum()
# 33 flagged

'''

# Plot for test cases, mark anomaly_check2 in red
anomalies_check = anomalies.loc[anomalies['anomaly_check']==1, ['ds', 'fact']].set_index('ds')
anomalies_check = anomalies_check[anomalies_check.index > train_last]
anomalies_check.head()
connects_min.head()


# We will highlight the evening_picks
# syntax based on: https://stackoverflow.com/questions/48973471/how-to-highlight-weekends-for-time-series-line-plot-in-python


def find_weekend_indices(ds_indexed):
    ds_unindexed = ds_indexed.reset_index()
    datetime_array = pd.to_datetime(ds_unindexed['index'])
    indices = []
    for i in range(len(datetime_array)):
        if datetime_array[i].weekday() >= 5:
            indices.append(i)
    return indices

def find_occupied_hours(ds_indexed):
    ds_unindexed = ds_indexed.reset_index()
    datetime_array = pd.to_datetime(ds_unindexed['index'])
    indices = []
    for i in range(len(datetime_array)):
        if datetime_array[i].hour == 19:
            if datetime_array[i].minute >= 0 and datetime_array[i].minute <= 40:
                indices.append(i)
    return indices

def highlight_datetimes(indices, df):
    i = 0
    while i < len(indices)-1:
        plt.axvspan(df.index[indices[i]], df.index[indices[i] + 1], facecolor='green', edgecolor='none', alpha=.1)
        i += 1

# plot with lines with highlights
fig, ax = plt.subplots(figsize=(25, 10))
ax = plt.plot(connects_min[-num_points_to_pred:].index, connects_min[-num_points_to_pred:].connects, label = 'Observed no of connections')
ax = plt.scatter(anomalies_check.index,anomalies_check.fact,c='red', label = 'Anomaly')
# find to be highlighted areas
weekend_indices = find_weekend_indices(connects_min[-num_points_to_pred:])
occupied_indices = find_occupied_hours(connects_min[-num_points_to_pred:])
# highlight areas
#highlight_datetimes(weekend_indices, connects_min[-num_points_to_pred:])
highlight_datetimes(occupied_indices, connects_min[-num_points_to_pred:])
# formatting
plt.title('Number of connections per minute with anomalies identified based on a forecast', fontsize=20)
plt.legend(fontsize=15)
dates_rng = pd.date_range(connects_min[-num_points_to_pred:].index[0], connects_min[-num_points_to_pred:].index[-1], freq='4H')
plt.xticks(dates_rng, [dtz.strftime('%Y-%m-%d %H:%M') for dtz in dates_rng], rotation=85)
fig.savefig('C://Users/klubi/Desktop/learning/Python/Github/Time series/outputs/anomalies_test_line.png')

# plot with lines
plt.rcParams['figure.figsize']=25,10
plt.plot(connects_min[-num_points_to_pred:].index, connects_min[-num_points_to_pred:].connects, label = 'Observed no of connections')
plt.scatter(anomalies_check.index,anomalies_check.fact,c='red', label = 'Anomaly')
plt.title('Number of connections per minute with anomalies identified based on a forecast', fontsize=20)
plt.legend(fontsize=15)
# format axis tick labels
dates_rng = pd.date_range(connects_min[-num_points_to_pred:].index[0], connects_min[-num_points_to_pred:].index[-1], freq='4H')
plt.xticks(dates_rng, [dtz.strftime('%Y-%m-%d %H:%M') for dtz in dates_rng], rotation=45)
plt.savefig('C://Users/klubi/Desktop/learning/Python/Github/Time series/outputs/anomalies_test_line.png')


# scatter
plt.rcParams['figure.figsize']=25,10
ax1=plt.scatter(connects_min[-num_points_to_pred:].index, connects_min[-num_points_to_pred:].connects, s=5, label = 'Observed no of connections')
ax2=plt.scatter(anomalies_check.index,anomalies_check.fact,c='red', label = 'Anomaly')
# formatting
plt.xlim(connects_min[-num_points_to_pred:].index[0], connects_min[-num_points_to_pred:].index[-1])
#ax.axvspan('19:00', '19:40', color='red', alpha=0.3)
plt.title('Number of connections per minute (test dataset) with anomalies marked in red', fontsize=20)
plt.legend(fontsize=15)
dates_rng = pd.date_range(connects_min[-num_points_to_pred:].index[0], connects_min[-num_points_to_pred:].index[-1], freq='4H')
plt.xticks(dates_rng, [dtz.strftime('%Y-%m-%d %H:%M') for dtz in dates_rng], rotation=90)
highlight_weekends(ax, connects_min[-num_points_to_pred:].index)
ax.savefig('C://Users/klubi/Desktop/learning/Python/Github/Time series/outputs/anomalies_test_scatter.png')
