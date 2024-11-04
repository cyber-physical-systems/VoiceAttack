import pandas as pd
import os
import csv
import glob

input_path = ''
output_path = '/1min/'


for name in glob.glob(input_path + '*.csv'):
    file_name = name.split('/')[-1]
#     df = pd.read_csv(name,index_col=[0])
    df = pd.read_csv(name)
    df['TIME'] = pd.to_datetime(df['TIME'], format='%Y-%m-%d  %H:%M:%S')
    df.index = df['TIME'] #change time to index
    df = df.drop('TIME', axis=1)#change time to index
    df = df.resample('1min').sum()
    df.to_csv(output_path + file_name, header=True)