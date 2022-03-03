import numpy as np
import pandas as pd
from tqdm import tqdm
import os


'''A Basic LSTM model for baseline comparisons'''



def load_data(experiment_name, root, time_scale, data_file):

    #Create requisite directory structure
    outdir = root+'/'+time_scale+'/'+experiment_name
    if not os.path.exists(outdir):
        os.makedirs(outdir,exist_ok=True)


    #Load 'time_scale' resolution data
    #Volume is by ETH and not USD (USD volume is dropped)
    #Args are used to specifically read dat column as a DateTimeIndex
    data = pd.read_csv(root+'/'+data_file,index_col=0,parse_dates=True)
    if(data_file == 'ETH_day.csv'):
        data = data.drop(['Symbol', 'Volume USD'], axis=1)
        data = data.rename(columns={'Volume ETH':'Volume'})
    else:
        print(data.head())
        data = data.reset_index()
        data = data.drop(['Symbol','Unix Timestamp'], axis=1)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index(['Date'])
    data = data.iloc[::-1]
    print(data.head())



    #labels.to_csv(root+'/'+time_scale+'/'+experiment_name+'/labels.csv')


load_data(experiment_name='basic_lstm',
          root='./data',
          time_scale='hourly',
          data_file='ETH_1H.csv')

