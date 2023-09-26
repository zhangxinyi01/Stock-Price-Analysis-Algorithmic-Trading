import yfinance as yf
import pandas as pd
import numpy as np
import sqlalchemy
import pymysql
from sqlalchemy import text

def storage(tickers_list, weight, db):
    # check if each element exsits in the api database:
    df = pd.read_excel('symbol_response.xlsx')
    for item in tickers_list:
        temp = df.loc[df[0] == item]
        if temp[1].values == "Series([], Name: Adj Close, dtype: float64)" or item not in temp[0].values:
            print(f'Stock symbol {item} not found, please try again.')
            return
    # Fetch the data
    print('Downloading...')
    closeData = yf.download(tickers_list,period='2y')['Adj Close']
    print('Null analysis of close stock price:\n', closeData.isnull().sum())
    
    changeData = closeData.pct_change()
    # calculate metrics for the portfolio: 
    # Calculate portfolio returns
    changeData['Portfolio'] = np.dot(changeData, weight)
    # Calculate portfolio standard deviation (risk)
    portfolio_std_dev = changeData['Portfolio'].std()
    # Define the risk-free rate (e.g., Treasury yield)
    risk_free_rate = 0.01  # Replace with the appropriate risk-free rate
    # Calculate portfolio excess return
    portfolio_excess_return = changeData['Portfolio'] - risk_free_rate
    portfolio_adj_mean_return = portfolio_excess_return.mean()
    # Calculate Sharpe Ratio
    sharpe_ratio = (portfolio_excess_return.mean() / portfolio_std_dev) * np.sqrt(252)  # Assuming 252 trading days in a year
    portfolio_metrics = ['return_std','return_mean','sharpe_ratio']
    portfolio_metrics_values = [portfolio_std_dev, portfolio_adj_mean_return, sharpe_ratio]
    metricsData = pd.DataFrame(list(zip(portfolio_metrics, portfolio_metrics_values)), columns = ['metrics', 'value'])
    
    pymysql.install_as_MySQLdb()
    engine = sqlalchemy.create_engine("mysql+mysqldb://root:Dsci-560@localhost/yahoo_finance")
    #with engine.connect() as conn:
        #conn.execute(text("DROP DATABASE if exists portfolio1"))
        #conn.execute(text("CREATE DATABASE portfolio1")) #create db
    tickers_table = pd.DataFrame(tickers_list, columns = ['stock_name'])
    # used as log file
    tickers_table.to_sql(f'{db}_content', engine, if_exists = 'append', index=True)  
    closeData.to_sql(f'{db}_history', engine, if_exists = 'replace', index=True)
    metricsData.to_sql(f'{db}_metrics', engine, if_exists = 'replace', index=True)
    print(metricsData)

if __name__ == "__main__":
    db = input('Please enter name of your portfolio: ')
    tickers = input('Please enter name of your intended stocks (Use comma and space to seperate your symbols. Sample input: AAPL, WMT, IBM, META, AMZN, GOOG):\n')
    weightInput = input("Please type in the weight for renewed each stock (Use comma and space to separate. Sample input: 0.3, 0.2, 0.2, 0.1, 0.2):\n")
    weight = weightInput.split(", ")
    weight = [float(i) for i in weight]
    tickers_list = tickers.split(", ")
    storage(tickers_list, weight, db)




