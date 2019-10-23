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
from sklearn.metrics import mean_absolute_error as mae
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from fbprophet import Prophet
from fbprophet.plot import plot
import holidays
from datetime import datetime
from joblib import dump, load

# Settings 
sns.set()
%matplotlib inline

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.plotting.register_matplotlib_converters()

# Change working directory
os.getcwd()
os.chdir(".../Time series/dane/all")
os.getcwd()

### DATA EXPLORATION

# Load and explore pickled data
connects_min = pd.read_pickle("connects_min.pkl")

connects_min.info()
connects_min.head()

# Check if it is possible to reduce the size of the file by changing the data type
connects_min['connects'].describe().apply(lambda x: format(x, 'f'))
# max = 69578. 
int_types = ["uint8", "uint16", "uint32"]
for it in int_types:
    print(np.iinfo(it))

# change to uint32
connects_min = pd.DataFrame(connects_min['connects'].astype('uint32'))
connects_min.info()
# memory usage reduced from 495.0 KB to 371.2 KB. 

### PRE-MODELLING DATA ANALYSIS

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

# select 3 different days to zoom in
connects_min[connects_min.index.day==2].plot(figsize=(16,9))
plt.title('Day: {}'.format(connects_min[connects_min.index.day==2].index[0].weekday_name))
plt.show();

connects_min[connects_min.index.day==4].plot(figsize=(16,9))
plt.title('Day: {}'.format(connects_min[connects_min.index.day==4].index[0].weekday_name))
plt.show();

connects_min[connects_min.index.day==7].plot(figsize=(16,9))
plt.title('Day: {}'.format(connects_min[connects_min.index.day==7].index[0].weekday_name))
plt.show();

'''
There seem to be:
- increased activity during working hours (week days)
- reduced activity during weekends
- short peaks around 7pm (also during weekends)
- short drops around 10pm (also during weekends)
- slightly downward trend

Prophet is framing the forecasting problem as a curve-fitting exercise rather than looking explicitly at the time based dependence of each observation within a time series. The observations above can be translated to the following assumptions and model parametrisation:
 - there is a minimum achieavable point due to contraints = 0, hence the 'floor' parameter should be set to 0. Domain expert says there is no saturating maximum, but to use a logistic growth trend with a saturating minimum, a maximum capacity must also be specified. I will check for the max historical value;
 - it can be seen from the plot that there is roughly constant level (the mean of 19942 connections per minute), no changepoints are visible. The seasonal fluctuation and random fluctuations roughly are constant in size over time. This suggests that it’s probably appropriate to describe the data using an additive model which is Prophet built on,
 - holidays will have an effect on the timeseries (as can be concluded based on reduced activity during the weekends). Although I poses of just a few weeks of data at the moment, I will account for holidays by using a built-in collection of country-specific holidays,
- seasonality_mode: default value here is additive with multiplicative being the other option. I will use additive seing that the seasonality trends (both daily and weekly) is “constant” over the entire period.
- seasonality_prior_scale parameter. This parameter allows seasonalities to be more flexible (sometimes smaller, sometimes bigger effect). 

'''

### PREPARE TRAINING AND TEST DATASETS 

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

### DATA MODELLING 

# Prepare data for fbprohpet

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

# Step 1. Start with weekly_seasonality, daily_seasonality and yearly_seasonality.
m1 = Prophet(growth='logistic', 
             daily_seasonality = True, 
             weekly_seasonality = True, 
             yearly_seasonality = True)
m1.add_country_holidays(country_name= 'PL')
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
    """Plot observed data from test dataset vs forecast, save the plot 
    in the specified folder and annotate it with Mean Absolute Error.
    Repeat the plot using fbprophpet library (incl. confidence intervals). 
    Save a plot of time series components (decomposition)."""
    
    forecast = forecast[-num_points_to_pred:]
    score = np.round(mae(test.y, forecast.yhat),0)
    plt.rcParams['figure.figsize']=15,10
    plt.title('Forecast last {} points (minutes).\nmae={}\nProphet'.format(num_points_to_pred, score))
    plt.plot(test.ds, test.y, label='test: observed')
    plt.plot(test.ds, forecast.yhat, label='forecast')
    plt.legend();
    plt.savefig('.../Time series/outputs/{}_mae={} .png'.format(model_string, score))
    
    m.plot(forecast)
    
    m.plot_components(forecast).savefig('.../Time series/outputs/{}_components_mae={} .png'.format(model_string, score));

# Evaluate and plot
plot_forecast(fb_test, forecast, 'm1', m1)

'''
MAE = 3502. 
Shortcommings of the first model (based on the forecast plot):
- underestimated activity during working hours at week days: daily seasonality needs to be made more flexible, use seasonality_prior_scale parameter,
- overestimated activity during weekends: model unnecessarily assumes increased activity during working hours: need to account for weekend seasonality
- short large peaks around 7pm (also during weekends): model underestimates them: again, seasonality_prior_scale parameter, 
the seasonality needs to fit higher-frequency changes, and generally be less smooth. The Fourier order can be specified for each built-in seasonality when instantiating the model, i.e. daily_seasonality = ... e.g. 40 (default=10)
- short drops around 10pm (also during weekends) - underestimated

Looking at the components plot:
- Weekly trend apparently detected - weekend drop can be observed,
- Daily trend is definitely needed, but, as mentioned above, needs to be more sensitive to big spikes/ drops. Also, we need a different daily trend for weekdays and weekends. These types of seasonalities can be modeled using conditional seasonalities. By doing this you get more power and control over seasonality.
'''

# Step 2. Set a daily seasonal pattern that is different on weekends vs. on weekdays. 

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


# Disable the built-in daily seasonality, and replace it with two daily seasonalities that have these columns specified as a condition. This means that the seasonality will only be applied to dates where the condition_name column is True. 
m2 = Prophet(growth='logistic', 
             daily_seasonality = False, 
             weekly_seasonality = True, 
             yearly_seasonality = True)
m2.add_country_holidays(country_name= 'DK')
m2.add_seasonality(name='daily_weekdays', 
                   period=1, # number of days in one period 
                   fourier_order=100, # fit higher-frequency changes
                   prior_scale=100, # allow more flexibility, need to detect daily peaks and dumps better  
                   condition_name='week_days')
m2.add_seasonality(name='daily_weekends', 
                   period=1,
                   fourier_order=100,
                   prior_scale=100,
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

# A significant imporvement is observed. MAE decreased from 3502 to 2046. 

# zoom in
plt.rcParams['figure.figsize']=25,10
plt.title('Forecast last {} points (minutes)'.format(num_points_to_pred))
plt.plot(fb_test.ds, fb_test.y, label='test: observed')
plt.plot(fb_test.ds, forecast2[-num_points_to_pred:].yhat, label='forecast')
plt.legend();

# Step 3. Add hourly seasonality
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

# A significant imporvement is observed. MAE decreased from 2046 to 1901. 

m3.plot(forecast3)

# zoom in
plt.rcParams['figure.figsize']=25,10
plt.title('Forecast last {} points (minutes)'.format(num_points_to_pred))
plt.plot(fb_test.ds, fb_test.y, label='test: observed')
plt.plot(fb_test.ds, forecast3[-num_points_to_pred:].yhat, label='forecast')
plt.legend();

# Evenings peaks are not fully covered even if we increase the pror_scale parameter (tried multiple values). Maybe introducing an additional effect (regressor) will help?

# Zoom in to investigate the evening pick
# week day
connects_min['2019-09-04 18:00':'2019-09-04 20:00'].plot(figsize=(16,9))
plt.title('Week day: 2019-09-04 18:00 - 2019-09-04 20:00')
plt.show();
# pick between 19:00 and 19:40

# weekend
connects_min['2019-09-15 18:00':'2019-09-15 20:00'].plot(figsize=(16,9))
plt.title('Weekend: 2019-09-15 18:00 - 2019-09-15 20:00')
plt.show();
# pick between 19:00 and 19:40

# Step 4: Add a regressor to improve detection of evening peaks (after-work report generation)

# Additional regressors can be added to the linear part of the model using the add_regressor method or function. A column with the regressor value will need to be present in both the fitting and prediction dataframes. The add_regressor function has optional argument for specifying the prior scale, by default this parameter is 10, which provides very little regularization. Reducing this parameter dampens the regressor's effect. 

# from datetime import datetime
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
                   period=1,
                   fourier_order=100,
                   prior_scale=100,
                   condition_name='week_days')
m4.add_seasonality(name='daily_weekends', 
                   period=1,
                   fourier_order=100,
                   prior_scale=100,
                   condition_name='weekends')
m4.add_seasonality(name='hourly', 
                   period=1/24, # day/number of hours per day
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

# Slight imporvement is observed. MAE decreased from 1901 to 1895. 
# After checking results for different prior_scale levels of .add_regressor function, the defauly values offered the only solution that reduces MAE. However, we are still not able to predict the highest picks. In order to avoid detecting all of them as outliers, we will change the uncertainty interval from the deafult .80 to 0.99 to produce wider confidence intervals around the forecasted values. 

# Step 5: Increase confidence interval around the prediction
m5 = Prophet(growth='logistic', 
             daily_seasonality = False, 
             weekly_seasonality = True, 
             yearly_seasonality = True, 
             interval_width=0.99) # increased from 0.80 to 0.99
m5.add_country_holidays(country_name= 'DK')
m5.add_seasonality(name='daily_weekdays', 
                   period=1,
                   fourier_order=100,
                   prior_scale=100,  
                   condition_name='week_days')
m5.add_seasonality(name='daily_weekends', 
                   period=1,
                   fourier_order=100,
                   prior_scale=100,
                   condition_name='weekends')
m5.add_seasonality(name='hourly', 
                   period=1/24,
                   fourier_order=30,
                   prior_scale=70)
m5.add_regressor('evening_pick')

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

### SAVE FINAL MODEL
# use joblib’s replacement of pickle (dump & load), which is more efficient on objects that carry large numpy arrays internally
# from joblib import dump, load

dump(m5, 'model5.joblib') 
