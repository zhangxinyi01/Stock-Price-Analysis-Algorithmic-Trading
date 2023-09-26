import yfinance as yf
import pandas as pd

df = pd.read_excel("stocksymbol.xlsx")

tickers_list = df.values.flatten().tolist()
# print(tickers_list)
# Fetch the data
import yfinance as yf
data = []
cnt = 0
for i in tickers_list:
    cnt = cnt + 1
    if cnt % 100 ==0:
        print(cnt)
    try:
        data.append([i, yf.download(i,period='6mo')['Adj Close']])
        # print(data)
    except:
        print("Error")

# Print first 5 rows of the data
# print(data.info)

data = pd.DataFrame(data)
data.to_excel("symbol_response.xlsx")
