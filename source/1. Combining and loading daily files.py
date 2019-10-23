### Data processing
### Author: Klaudia Łubian - StatsEye
### Date: 23/10/2019

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

'''
The data on TCP connections was collected daily (a string with date and time marking a connection made)
and saved in the working directory. 
Now it needs to be loaded and combined into a time series of multiple weeks.  Size of files ranges from 400 - 700 MB, 
hence operations on the combined dataset will not be manageable for computer's memory. 
I want to create a summary file with counts of signals per minute. A solution to this is to process an entire data source chunk by chunk, 
instead of a single go all at once. Therefore:
1) I will apply glob() function to read multiple files (they have similar names)
2) The data will be processes chunk by chunk using chunksize in pd.read_csv(), what will return a dictionary with counts of
   occurrences as value for each key (datetime).
'''

# Write the pattern for detecting files of interest in WD:
pattern = '2019*'

# Save all file matches
txt_files = glob.glob(pattern)

# Print the file names
print(txt_files)

# Define count_entries() fuction 
# Files include only one column: a string with date and time marking a connection made

def count_entries(csv_file, c_size, colname, pattern):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Iterate over the file chunk by chunk, parse dates
    for chunk in pd.read_csv(csv_file, 
                             chunksize= c_size, 
                             names = [colname],
                             parse_dates= [colname],
                             date_parser = lambda dates: [pd.datetime.strptime(d, pattern) for d in dates]):
        # Sort by DatetimeIndex
        # chunk = chunk.sort_values([colname], ascending=[True])
          
        # Iterate over the column in DataFrame
        for entry in chunk[colname]:
            # If date is already in dictionary, add one, if not = 1
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1

    # Return counts_dict
    return counts_dict

# Create a list of dictionaries, each dictionary containing counts from one dataset. Iterate over txt_files and apply count_entries() function. 
all_counts = []
for txtf in txt_files:
    
    print(txtf)
    counts = count_entries(txtf, 100000, 'date', '%Y-%m-%dT%H:%M:%S')
    all_counts.append(counts)
    
# Check number of dictionaries within a list built
len(all_counts)

# Create DataFrame from list of dictionaries
df_sec = pd.DataFrame(all_counts).T
df_sec.tail()
df_sec.head()
df_sec['connects'] = df_sec.fillna(0).sum(axis=1)
df_sec.head()
df_sec.tail()
df_sec.info()
# We could reduce the memory usage by changing the type of the variables. Now it's float64, memory usage: 347.9 MB. 
# check min and max of the number of connections, suppress scientific notation (function describe returns a dataframe)
df_sec['connects'].describe().apply(lambda x: format(x, 'f'))
# max = 9946. 
int_types = ["uint8", "int8", "uint16", "int16"]
for it in int_types:
    print(np.iinfo(it))

# uint16 would be enough. Keep just one column
connects_sec = pd.DataFrame(df_sec['connects'].astype('uint16'))
connects_sec.info()
# memory usage: 18.1 MB
connects_sec['connects'].describe().apply(lambda x: format(x, 'f'))
connects_sec.head()   

# Sort dataframe 
connects_sec = connects_sec.sort_index()
connects_sec.info()
# time frame 2019-09-01 22:00:03 to 2019-09-23 21:59:59
# Store dataframe
connects_sec.to_pickle("connects_sec.pkl")


# Resample the data at a different frequency
# Downsample df_sec into 1 minute bins and sum the values of the timestamps falling into a bin.

connects_min = connects_sec.resample('1T').sum()
connects_min.head()
connects_min.info()

# Store dataframe
connects_min.to_pickle("connects_min.pkl")

# Plot data
connects_min.plot(figsize=(20,8))
plt.show()
