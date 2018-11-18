import pandas as pd
import numpy as np
from datetime import date

# Read Meta Data file and Test/Tarin data file
df_meta              = pd.read_csv("..\\data\\meta.csv")
df_cold_start        = pd.read_csv("..\\data\\cold_start_test.csv")
df_cold_start_unique = np.unique(df_cold_start['series_id'].values)
store                = pd.HDFStore('submission_data.h5')

# Add new columns to data frame to add meta data
df_cold_start               = df_cold_start.assign(IS_HOLIDAY=False, surface=0, base_temperature=0, time=0, day=0)
df_meta['surface']          = df_meta['surface'].astype('category')
df_meta['surface']          = df_meta['surface'].cat.codes
df_meta['base_temperature'] = df_meta['base_temperature'].astype('category')
df_meta['base_temperature'] = df_meta['base_temperature'].cat.codes

# Directy copying series id secific data.
for i in df_cold_start_unique:
    index_list = df_cold_start.loc[df_cold_start['series_id']==i].index
    df_cold_start.loc[index_list,'surface'] = df_meta[df_meta['series_id']==i]['surface'].values
    df_cold_start.loc[index_list,'base_temperature'] = df_meta[df_meta['series_id']==i]['base_temperature'].values

# Add time and date to Dataframe
for i in range(df_cold_start.shape[0]):    
    a = df_cold_start.loc[i, 'timestamp']
    [date_new, time] = a.split(" ")
    [year, month, day] = date_new.split("-")
    [time, x1, x2] = time.split(":")
    df_cold_start.loc[i, 'time'] = int(time)
    current_weekday = int(date(int(year), int(month), int(day)).weekday())
    df_cold_start.loc[i, 'day']  = current_weekday
    if (i%100) == 0:
        print(i)

# To add holiday need to iterate over all days of each series_id
start_index = df_meta.columns.get_loc("monday_is_day_off")

for i in range(df_cold_start.shape[0]):
    index = df_meta.loc[df_meta['series_id']==df_cold_start.loc[i,'series_id']].index
    df = df_meta.loc[index]
    df_cold_start.loc[i, 'IS_HOLIDAY'] = df.iloc[0, start_index + df_cold_start.loc[i, 'day']]
    if (i%100) == 0:
        print(i)

# Save Newly Created Dataframe
store['df_cold_start'] = df_cold_start    
