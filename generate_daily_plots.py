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
    price_change_labels = []
    #Label each image based on percent price change
    percent_change_labels = []
    #Label for current day price
    cur_price_labels = []
    #Label for the resulting price in the next time step
    result_price_labels = []


    #Plot day-by-day candle-stick chart, with previous 'time_window' days history, for every day
    #We do not include the very last day because we do not know its price change in the following time step.
    for i in tqdm(range(day_data.shape[0]-time_window-1)):
        current_timeStep = day_data.iloc[i:i+time_window]
        mpf.plot(current_timeStep, type='candle', volume=True, style='yahoo', savefig=root+'/daily/candle_stick/timestep_'+str(i))

        # Initialise t+1 price for label generation
        resulting_price = day_data.iloc[i+(time_window + 1)]['Close']

        #The label will be the net change in price at closing time from the current time step to the next
        cur_price = day_data.iloc[i+time_window]['Close']
        price_change = resulting_price - cur_price
        price_change_labels.append(price_change)
        percent_change_labels.append(price_change / cur_price)
        cur_price_labels.append(cur_price)
        result_price_labels.append(resulting_price)


    labels = pd.DataFrame(price_change_labels)
    labels.columns = ['closing_price_change_usd']
    labels['percent_change'] = percent_change_labels
    labels['current_price'] = cur_price_labels
    labels['resulting_price'] = result_price_labels

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
    price_change_labels = []
    #Label each image based on percent price change
    percent_change_labels = []
    #Label for current day price
    cur_price_labels = []
    #Label for the resulting price in the next time step
    result_price_labels = []


    #Plot day-by-day candle-stick chart, with previous 'time_window' days history, for every day
    #We do not include the very last day because we do not know its price change in the following time step.
    for i in tqdm(range(day_data.shape[0]-time_window-1)):
        current_timeStep = day_data.iloc[i:i+time_window]
        mpf.plot(current_timeStep, type='line', volume=True, style='yahoo', savefig=root+'/daily/price_line/timestep_'+str(i))

        # Initialise t+1 price for label generation
        resulting_price = day_data.iloc[i+(time_window + 1)]['Close']

        #The label will be the net change in price at closing time from the current time step to the next
        cur_price = day_data.iloc[i+time_window]['Close']
        price_change = resulting_price - cur_price
        price_change_labels.append(price_change)
        percent_change_labels.append(price_change / cur_price)
        cur_price_labels.append(cur_price)
        result_price_labels.append(resulting_price)


    labels = pd.DataFrame(price_change_labels)
    labels.columns = ['closing_price_change_usd']
    labels['percent_change'] = percent_change_labels
    labels['current_price'] = cur_price_labels
    labels['resulting_price'] = result_price_labels

    labels.to_csv(root+'/daily/price_line/labels.csv')



def gen_day_PandF(root, time_window):

    #Create requisite directory structure
    outdir = root+'/daily/PandF'
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
    price_change_labels = []
    #Label each image based on percent price change
    percent_change_labels = []
    #Label for current day price
    cur_price_labels = []
    #Label for the resulting price in the next time step
    result_price_labels = []


    #Plot day-by-day point and figure chart, with previous 'time_window' days history, for every day
    #We do not include the very last day because we do not know its price change in the following time step.
    for i in tqdm(range(day_data.shape[0]-time_window-1)):
        current_timeStep = day_data.iloc[i:i+time_window]
        mpf.plot(current_timeStep, type='pnf', style='yahoo', savefig=root+'/daily/PandF/timestep_'+str(i))

        # Initialise t+1 price for label generation
        resulting_price = day_data.iloc[i+(time_window + 1)]['Close']

        #The label will be the net change in price at closing time from the current time step to the next
        cur_price = day_data.iloc[i+time_window]['Close']
        price_change = resulting_price - cur_price
        price_change_labels.append(price_change)
        percent_change_labels.append(price_change / cur_price)
        cur_price_labels.append(cur_price)
        result_price_labels.append(resulting_price)


    labels = pd.DataFrame(price_change_labels)
    labels.columns = ['closing_price_change_usd']
    labels['percent_change'] = percent_change_labels
    labels['current_price'] = cur_price_labels
    labels['resulting_price'] = result_price_labels

    labels.to_csv(root+'/daily/PandF/labels.csv')


def gen_day_renko(root, time_window):

    #Create requisite directory structure
    outdir = root+'/daily/renko'
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

    # Label each image based on closing price increase/drop for the given time step
    price_change_labels = []
    # Label each image based on percent price change
    percent_change_labels = []
    # Label for current day price
    cur_price_labels = []
    # Label for the resulting price in the next time step
    result_price_labels = []


    #Plot day-by-day renko chart, with previous 'time_window' days history, for every day
    #We do not include the very last day because we do not know its price change in the following time step.
    for i in tqdm(range(day_data.shape[0]-time_window-1)):
        current_timeStep = day_data.iloc[i:i+time_window]
        mpf.plot(current_timeStep, type='renko', style='yahoo', savefig=root+'/daily/renko/timestep_'+str(i))

        # Initialise t+1 price for label generation
        resulting_price = day_data.iloc[i+(time_window + 1)]['Close']

        #The label will be the net change in price at closing time from the current time step to the next
        cur_price = day_data.iloc[i+time_window]['Close']
        price_change = resulting_price - cur_price
        price_change_labels.append(price_change)
        percent_change_labels.append(price_change / cur_price)
        cur_price_labels.append(cur_price)
        result_price_labels.append(resulting_price)


    labels = pd.DataFrame(price_change_labels)
    labels.columns = ['closing_price_change_usd']
    labels['percent_change'] = percent_change_labels
    labels['current_price'] = cur_price_labels
    labels['resulting_price'] = result_price_labels

    labels.to_csv(root+'/daily/renko/labels.csv')




def gen_day_movingAvg(root, time_window):

    #Create requisite directory structure
    outdir = root+'/daily/movingAvg'
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
    price_change_labels = []
    #Label each image based on percent price change
    percent_change_labels = []
    #Label for current day price
    cur_price_labels = []
    #Label for the resulting price in the next time step
    result_price_labels = []

    #Plot day-by-day candle-stick chart, with previous 'time_window' days history, for every day
    #We do not include the very last day because we do not know its price change in the following time step.
    for i in tqdm(range(day_data.shape[0]-time_window-1)):
        current_timeStep = day_data.iloc[i:i+time_window]

        #For moving averages, we can use aproximately 3/4, 1/2, and 1/4 of the chosen time period
        mav1 = int(np.floor(.75 * time_window))
        mav2 = int(np.floor(.5 * time_window))
        mav3 = int(np.floor(.25 * time_window))

        mpf.plot(current_timeStep, type='candle', mav=(mav1,mav2,mav3), savefig=root+'/daily/movingAvg/timestep_'+str(i))

        # Initialise t+1 price for label generation
        resulting_price = day_data.iloc[i+(time_window + 1)]['Close']

        #The label will be the net change in price at closing time from the current time step to the next
        cur_price = day_data.iloc[i+time_window]['Close']
        price_change = resulting_price - cur_price
        price_change_labels.append(price_change)
        percent_change_labels.append(price_change/cur_price)
        cur_price_labels.append(cur_price)
        result_price_labels.append(resulting_price)


    labels = pd.DataFrame(price_change_labels)
    labels.columns = ['closing_price_change_usd']
    labels['percent_change'] = percent_change_labels
    labels['current_price'] = cur_price_labels
    labels['resulting_price'] = result_price_labels

    labels.to_csv(root+'/daily/movingAvg/labels.csv')

'''###############################################'''
root = './data'
time_window=30
gen_day_candlestick(root,time_window)
gen_day_priceLine(root,time_window)
gen_day_PandF(root,time_window)
gen_day_renko(root,time_window)
gen_day_movingAvg(root,time_window)




