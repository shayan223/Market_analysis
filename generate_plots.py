import numpy as np
import mplfinance as mpf
import pandas as pd
from tqdm import tqdm
import os



def gen_candlestick(root, time_window, time_scale,data_file):

    #Create requisite directory structure
    outdir = root+'/'+time_scale+'/candle_stick'
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
        data = data.drop(['Symbol','Unix Timestamp'], axis=1)
    data = data.iloc[::-1]
    print(data.head())

    #Label each image based on closing price increase/drop for the given time step
    price_change_labels = []
    #Label each image based on percent price change
    percent_change_labels = []
    #Label for current day price
    cur_price_labels = []
    #Label for the resulting price in the next time step
    result_price_labels = []
    #Keep track of every file name
    file_name_labels = []


    #Plot day-by-day candle-stick chart, with previous 'time_window' days history, for every day
    #We do not include the very last day because we do not know its price change in the following time step.
    for i in tqdm(range(data.shape[0]-time_window-1)):
        current_timeStep = data.iloc[i:i+time_window]
        # Change size of saved figures to match resnet image resolution of 224x224 (default dpi of 100 used in calculation below)
        mpf.plot(current_timeStep, type='candle', volume=True, style='yahoo', savefig=root+'/'+time_scale+'/candle_stick/timestep_'+str(i),figsize=(224/100,224/100))


        # Initialise t+1 price for label generation
        resulting_price = data.iloc[i+(time_window + 1)]['Close']

        #The label will be the net change in price at closing time from the current time step to the next
        cur_price = data.iloc[i+time_window]['Close']
        price_change = resulting_price - cur_price
        price_change_labels.append(price_change)
        percent_change_labels.append(price_change / cur_price)
        cur_price_labels.append(cur_price)
        result_price_labels.append(resulting_price)

        filename = 'timestep_'+str(i)+'.png'
        file_name_labels.append(filename)


    labels = pd.DataFrame(price_change_labels)
    labels.columns = ['closing_price_change_usd']
    labels['percent_change'] = percent_change_labels
    labels['current_price'] = cur_price_labels
    labels['resulting_price'] = result_price_labels
    labels['file_name'] = file_name_labels

    labels.to_csv(root+'/'+time_scale+'/candle_stick/labels.csv')




def gen_priceLine(root, time_window,time_scale,data_file):

    #Create requisite directory structure
    outdir = root+'/'+time_scale+'/price_line'
    if not os.path.exists(outdir):
        os.makedirs(outdir,exist_ok=True)


    #Load day-by-day resolution data
    #Volume is by USD and not ETH (ETH volume is dropped)
    #Args are used to specifically read dat column as a DateTimeIndex
    data = pd.read_csv(root+'/'+data_file,index_col=0,parse_dates=True)
    data = data.drop(['Symbol', 'Volume ETH'], axis=1)
    data = data.rename(columns={'Volume USD':'Volume'})
    data = data.iloc[::-1]
    print(data.head())

    #Label each image based on closing price increase/drop for the given time step
    price_change_labels = []
    #Label each image based on percent price change
    percent_change_labels = []
    #Label for current day price
    cur_price_labels = []
    #Label for the resulting price in the next time step
    result_price_labels = []
    #Keep track of every file name
    file_name_labels = []


    #Plot day-by-day candle-stick chart, with previous 'time_window' days history, for every day
    #We do not include the very last day because we do not know its price change in the following time step.
    for i in tqdm(range(data.shape[0]-time_window-1)):
        current_timeStep = data.iloc[i:i+time_window]
        # Change size of saved figures to match resnet image resolution of 224x224 (default dpi of 100 used in calculation below)
        mpf.plot(current_timeStep, type='line', volume=True, style='yahoo', savefig=root+'/'+time_scale+'/price_line/timestep_'+str(i),figsize=(224/100,224/100))

        # Initialise t+1 price for label generation
        resulting_price = data.iloc[i+(time_window + 1)]['Close']

        #The label will be the net change in price at closing time from the current time step to the next
        cur_price = data.iloc[i+time_window]['Close']
        price_change = resulting_price - cur_price
        price_change_labels.append(price_change)
        percent_change_labels.append(price_change / cur_price)
        cur_price_labels.append(cur_price)
        result_price_labels.append(resulting_price)

        filename = 'timestep_'+str(i)+'.png'
        file_name_labels.append(filename)


    labels = pd.DataFrame(price_change_labels)
    labels.columns = ['closing_price_change_usd']
    labels['percent_change'] = percent_change_labels
    labels['current_price'] = cur_price_labels
    labels['resulting_price'] = result_price_labels
    labels['file_name'] = file_name_labels

    labels.to_csv(root+'/'+time_scale+'/price_line/labels.csv')



def gen_PandF(root, time_window,time_scale,data_file):

    #Create requisite directory structure
    outdir = root+'/'+time_scale+'/PandF'
    if not os.path.exists(outdir):
        os.makedirs(outdir,exist_ok=True)


    #Load day-by-day resolution data
    #Volume is by USD and not ETH (ETH volume is dropped)
    #Args are used to specifically read dat column as a DateTimeIndex
    data = pd.read_csv(root+'/'+data_file,index_col=0,parse_dates=True)
    data = data.drop(['Symbol', 'Volume ETH'], axis=1)
    data = data.rename(columns={'Volume USD':'Volume'})
    data = data.iloc[::-1]
    print(data.head())

    #Label each image based on closing price increase/drop for the given time step
    price_change_labels = []
    #Label each image based on percent price change
    percent_change_labels = []
    #Label for current day price
    cur_price_labels = []
    #Label for the resulting price in the next time step
    result_price_labels = []
    #Keep track of every file name
    file_name_labels = []


    #Plot day-by-day point and figure chart, with previous 'time_window' days history, for every day
    #We do not include the very last day because we do not know its price change in the following time step.
    for i in tqdm(range(data.shape[0]-time_window-1)):
        current_timeStep = data.iloc[i:i+time_window]
        # Change size of saved figures to match resnet image resolution of 224x224 (default dpi of 100 used in calculation below)
        mpf.plot(current_timeStep, type='pnf', style='yahoo', savefig=root+'/'+time_scale+'/PandF/timestep_'+str(i),figsize=(224/100,224/100))

        # Initialise t+1 price for label generation
        resulting_price = data.iloc[i+(time_window + 1)]['Close']

        #The label will be the net change in price at closing time from the current time step to the next
        cur_price = data.iloc[i+time_window]['Close']
        price_change = resulting_price - cur_price
        price_change_labels.append(price_change)
        percent_change_labels.append(price_change / cur_price)
        cur_price_labels.append(cur_price)
        result_price_labels.append(resulting_price)

        filename = 'timestep_'+str(i)+'.png'
        file_name_labels.append(filename)


    labels = pd.DataFrame(price_change_labels)
    labels.columns = ['closing_price_change_usd']
    labels['percent_change'] = percent_change_labels
    labels['current_price'] = cur_price_labels
    labels['resulting_price'] = result_price_labels
    labels['file_name'] = file_name_labels

    labels.to_csv(root+'/'+time_scale+'/PandF/labels.csv')


def gen_renko(root, time_window,time_scale,data_file):

    #Create requisite directory structure
    outdir = root+'/'+time_scale+'/renko'
    if not os.path.exists(outdir):
        os.makedirs(outdir,exist_ok=True)


    #Load day-by-day resolution data
    #Volume is by USD and not ETH (ETH volume is dropped)
    #Args are used to specifically read dat column as a DateTimeIndex
    data = pd.read_csv(root+'/'+data_file,index_col=0,parse_dates=True)
    data = data.drop(['Symbol', 'Volume ETH'], axis=1)
    data = data.rename(columns={'Volume USD':'Volume'})
    data = data.iloc[::-1]
    print(data.head())

    # Label each image based on closing price increase/drop for the given time step
    price_change_labels = []
    # Label each image based on percent price change
    percent_change_labels = []
    # Label for current day price
    cur_price_labels = []
    # Label for the resulting price in the next time step
    result_price_labels = []
    #Keep track of every file name
    file_name_labels = []


    #Plot day-by-day renko chart, with previous 'time_window' days history, for every day
    #We do not include the very last day because we do not know its price change in the following time step.
    for i in tqdm(range(data.shape[0]-time_window-1)):
        current_timeStep = data.iloc[i:i+time_window]
        # Change size of saved figures to match resnet image resolution of 224x224 (default dpi of 100 used in calculation below)
        mpf.plot(current_timeStep, type='renko', style='yahoo', savefig=root+'/'+time_scale+'/renko/timestep_'+str(i),figsize=(224/100,224/100))

        # Initialise t+1 price for label generation
        resulting_price = data.iloc[i+(time_window + 1)]['Close']

        #The label will be the net change in price at closing time from the current time step to the next
        cur_price = data.iloc[i+time_window]['Close']
        price_change = resulting_price - cur_price
        price_change_labels.append(price_change)
        percent_change_labels.append(price_change / cur_price)
        cur_price_labels.append(cur_price)
        result_price_labels.append(resulting_price)

        filename = 'timestep_'+str(i)+'.png'
        file_name_labels.append(filename)

    labels = pd.DataFrame(price_change_labels)
    labels.columns = ['closing_price_change_usd']
    labels['percent_change'] = percent_change_labels
    labels['current_price'] = cur_price_labels
    labels['resulting_price'] = result_price_labels
    labels['file_name'] = file_name_labels

    labels.to_csv(root+'/'+time_scale+'/renko/labels.csv')




def gen_movingAvg(root, time_window,time_scale,data_file):

    #Create requisite directory structure
    outdir = root+'/'+time_scale+'/movingAvg'
    if not os.path.exists(outdir):
        os.makedirs(outdir,exist_ok=True)


    #Load day-by-day resolution data
    #Volume is by USD and not ETH (ETH volume is dropped)
    #Args are used to specifically read dat column as a DateTimeIndex
    data = pd.read_csv(root+'/'+data_file,index_col=0,parse_dates=True)
    data = data.drop(['Symbol', 'Volume ETH'], axis=1)
    data = data.rename(columns={'Volume USD':'Volume'})
    data = data.iloc[::-1]
    print(data.head())

    #Label each image based on closing price increase/drop for the given time step
    price_change_labels = []
    #Label each image based on percent price change
    percent_change_labels = []
    #Label for current day price
    cur_price_labels = []
    #Label for the resulting price in the next time step
    result_price_labels = []
    #Keep track of every file name
    file_name_labels = []

    #Plot day-by-day candle-stick chart, with previous 'time_window' days history, for every day
    #We do not include the very last day because we do not know its price change in the following time step.
    for i in tqdm(range(data.shape[0]-time_window-1)):
        current_timeStep = data.iloc[i:i+time_window]

        #For moving averages, we can use aproximately 3/4, 1/2, and 1/4 of the chosen time period
        mav1 = int(np.floor(.75 * time_window))
        mav2 = int(np.floor(.5 * time_window))
        mav3 = int(np.floor(.25 * time_window))
        # Change size of saved figures to match resnet image resolution of 224x224 (default dpi of 100 used in calculation below)
        mpf.plot(current_timeStep, type='candle', mav=(mav1,mav2,mav3), savefig=root+'/'+time_scale+'/movingAvg/timestep_'+str(i),figsize=(224/100,224/100))

        # Initialise t+1 price for label generation
        resulting_price = data.iloc[i+(time_window + 1)]['Close']

        #The label will be the net change in price at closing time from the current time step to the next
        cur_price = data.iloc[i+time_window]['Close']
        price_change = resulting_price - cur_price
        price_change_labels.append(price_change)
        percent_change_labels.append(price_change/cur_price)
        cur_price_labels.append(cur_price)
        result_price_labels.append(resulting_price)

        filename = 'timestep_'+str(i)+'.png'
        file_name_labels.append(filename)


    labels = pd.DataFrame(price_change_labels)
    labels.columns = ['closing_price_change_usd']
    labels['percent_change'] = percent_change_labels
    labels['current_price'] = cur_price_labels
    labels['resulting_price'] = result_price_labels
    labels['file_name'] = file_name_labels

    labels.to_csv(root+'/'+time_scale+'/movingAvg/labels.csv')

'''###############################################'''

root = './data'
time_window=30
'''
gen_candlestick(root,time_window,'daily','ETH_day.csv')
gen_priceLine(root,time_window,'daily','ETH_day.csv')
gen_PandF(root,time_window,'daily','ETH_day.csv')
gen_renko(root,time_window,'daily','ETH_day.csv')
gen_movingAvg(root,time_window,'daily','ETH_day.csv')
'''

gen_candlestick(root,time_window,'hourly','ETH_1H.csv')
gen_priceLine(root,time_window,'hourly','ETH_1H.csv')
gen_PandF(root,time_window,'hourly','ETH_1H.csv')
gen_renko(root,time_window,'hourly','ETH_1H.csv')
gen_movingAvg(root,time_window,'hourly','ETH_1H.csv')




