import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
df1 = pd.read_csv('arima_predictions_all_columns_7000.csv')
df2 = pd.read_csv('pred_res.csv')
# filer data for user input
portfolio = input('Please input stocks in your portfolio: (Use ", " as seperator)\n Sample input: AAPL, MSFT, GOOG ')
stocks = portfolio.split(", ")
stocks_actural = [i+'_actual' for i in stocks]
stocks_predict = [i+'_predicted' for i in stocks]
 
df = pd.DataFrame()
for j in range(len(stocks)):
    model1_predictions = df1[f'{stocks[j]}_predicted']
    model2_predictions = df2[f'{stocks[j]}']
    actual_values = df1[f'{stocks[j]}_actual']
    #df.rename(columns = {stocks[j]+'_actual':stocks[j]+'_value'}, inplace = True)
    mae_model1 = np.mean(np.abs(model1_predictions - actual_values))
    mae_model2 = np.mean(np.abs(model2_predictions - actual_values))
    # Determine weights based on accuracy (lower MAE implies higher weight)
    weight_model1 = 1 / (mae_model1 + 1e-6)  # Adding a small value to prevent division by zero
    weight_model2 = 1 / (mae_model2 + 1e-6)
    # Normalize the weights so that they sum to 1
    total_weight = weight_model1 + weight_model2
    weight_model1 /= total_weight
    weight_model2 /= total_weight
    # Merge the predictions using weighted averaging
    df[f'{stocks[j]}_value'] = (weight_model1 * model1_predictions) + (weight_model2 * model2_predictions)
    df[f'{stocks[j]}_amount'] = 0

# calculate metrics for each of the stocks
def calculate_rsi(data, window=7):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


initial_fund = input('Please input your initial amount of fund: ')
#allocation = input('Please input your weight: ')
df['total'] = initial_fund
df['rest'] = 0
stock_rsi = [i+'_RSI' for i in stocks]
stock_sig = [i+'_Signal' for i in stocks]
# buy/sell signals for each stock at different timestamps
for i in stocks:
    df[f'{i}_RSI'] = calculate_rsi(df[f'{i}_value'])
    # Define overbought and oversold thresholds
    overbought_threshold = 70
    oversold_threshold = 30
    # Generate buy and sell signals
    df[f'{i}_Signal'] = np.where(df[f'{i}_RSI'] < oversold_threshold, 1, 0)
    df[f'{i}_Signal'] = np.where(df[f'{i}_RSI'] > overbought_threshold, -1, df[f'{i}_Signal'])

buy = {}
sell = {}
for i in range(0, len(df)):
    stock_buy = []
    stock_sell = []
    for j in range(len(stocks)):
        if df.iloc[i][f'{stocks[j]}_Signal'] == 1:
            stock_buy.append(stocks[j])
        if df.iloc[i][f'{stocks[j]}_Signal'] == -1:
            stock_sell.append(stocks[j])
    buy[i] = stock_buy
    sell[i] = stock_sell
# 计算buy的股票的amount
for i in range(0, len(df)):
    rest = 0
    for j in range(len(stocks)):
        if len(buy[i]) != 0:
            if stocks[j] in buy[i]:
                try:
                    df[f'{stocks[j]}_amount'][i] = float(df['total'][i-1])/len(buy[i])//df[f'{stocks[j]}_value'][i]
                    rest += float(df[f'{stocks[j]}_amount'][i]*df[f'{stocks[j]}_value'][i])
                except:
                    df[f'{stocks[j]}_amount'][i] = initial_fund/len(buy[i])//df[f'{stocks[j]}_value'][i]
                    rest += float(df[f'{stocks[j]}_amount'][i]*df[f'{stocks[j]}_value'][i])    
        else:
            df[f'{stocks[j]}_amount'][i] = 0
    df['rest'][i] = float(df['total'][i])-float(rest)
# 计算整个portfolio的总价值     
for i in range(1, len(df)):
    total = 0
    for j in range(len(stocks)):
        total += float(df[f'{stocks[j]}_amount'][i-1]*df[f'{stocks[j]}_value'][i])
        #rest = float(df['total'][i-1]) - float(df[f'{stocks[j]}_amount'][i]*df[f'{stocks[j]}_value'][i])
    if total != 0:
        df['total'][i] = float(total) + df['rest'][i-1]
    else:
        df['total'][i] = initial_fund

print(df.iloc[:, :2*len(stocks)+1])
                           