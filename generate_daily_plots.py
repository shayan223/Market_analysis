import numpy as np
import mplfinance as mpf
import pandas as pd
from tqdm import tqdm
import os


def gen_day_candlestick(root, time_window):

    #Create requisite directory structure
    outdir = root+'/daily/candle_stick'
    if not os.path.exists(outdir):
        os.makedirs(outdir,exist_ok=True)


    #Load day-by-day resolution data
    #Volume is by USD and not ETH (ETH volume is dropped)
    #Args are used to specifically read dat column as a DateTimeIndex
    day_data = pd.read_csv(root+'/ETH_day.csv',index_col=0,parse_dates=True)
    day_data = day_data.drop(['Symbol', 'Volume ETH'], axis=1)
    day_data = day_data.rename(columns={'Volume USD':'Volume'})
    day_data = day_data.iloc[::-1]
    print(day_data.head())

    #Label each image based on closing price increase/drop for the given time step
    temp_labels = []


    #Plot day-by-day candle-stick chart, with previous 'time_window' days history, for every day
    #We do not include the very last day because we do not know its price change in the following time step.
    for i in tqdm(range(day_data.shape[0]-time_window-1)):
        current_timeStep = day_data.iloc[i:i+time_window]
        mpf.plot(current_timeStep, type='candle', volume=True, style='yahoo', savefig=root+'/daily/candle_stick/timestep_'+str(i))

        # Initialise t+1 price for label generation
        resulting_price = day_data.iloc[i+(time_window + 1)]['Close']

        #The label will be the net change in price at closing time from the current time step to the next
        cur_price = day_data.iloc[i+time_window]['Close']
        temp_labels.append(resulting_price - cur_price)


    labels = pd.DataFrame(temp_labels)
    labels.columns = ['closing_price__change_usd']
    labels.to_csv(root+'/daily/candle_stick/labels.csv')




def gen_day_priceLine(root, time_window):

    #Create requisite directory structure
    outdir = root+'/daily/price_line'
    if not os.path.exists(outdir):
        os.makedirs(outdir,exist_ok=True)


    #Load day-by-day resolution data
    #Volume is by USD and not ETH (ETH volume is dropped)
    #Args are used to specifically read dat column as a DateTimeIndex
    day_data = pd.read_csv(root+'/ETH_day.csv',index_col=0,parse_dates=True)
    day_data = day_data.drop(['Symbol', 'Volume ETH'], axis=1)
    day_data = day_data.rename(columns={'Volume USD':'Volume'})
    day_data = day_data.iloc[::-1]
    print(day_data.head())

    #Label each image based on closing price increase/drop for the given time step
    temp_labels = []


    #Plot day-by-day candle-stick chart, with previous 'time_window' days history, for every day
    #We do not include the very last day because we do not know its price change in the following time step.
    for i in tqdm(range(day_data.shape[0]-time_window-1)):
        current_timeStep = day_data.iloc[i:i+time_window]
        mpf.plot(current_timeStep, type='line', volume=True, style='yahoo', savefig=root+'/daily/price_line/timestep_'+str(i))

        # Initialise t+1 price for label generation
        resulting_price = day_data.iloc[i+(time_window + 1)]['Close']

        #The label will be the net change in price at closing time from the current time step to the next
        cur_price = day_data.iloc[i+time_window]['Close']
        temp_labels.append(resulting_price - cur_price)


    labels = pd.DataFrame(temp_labels)
    labels.columns = ['closing_price__change_usd']
    labels.to_csv(root+'/daily/price_line/labels.csv')



root = './data'

time_window=30
gen_day_candlestick(root,time_window)
gen_day_priceLine(root,time_window)







